// Persistent genuine OCS2 SLQ worker. One JSON request/response per line.
// The physical supervisor owns command expiry, hard limits and collision brakes.
#include "CameraMpcModel.h"
#include "AccelerationDynamics.h"
#include "CollisionCost.h"
#include "ObservedEnvironmentCost.h"
#include <sstream>
#include <iomanip>

struct BrakingCost final : StateCost {
  vector_t lower,upper,acceleration;
  int nu;
  BrakingCost(const CameraModel& model,const vector_t& limits):lower(model.lower),upper(model.upper),acceleration(limits.tail(6)),nu(limits.size()){}
  BrakingCost* clone() const override{return new BrakingCost(*this);}
  vector_t residual(const vector_t& x) const {
    vector_t r(12);
    for(int i=0;i<6;++i){
      double v=x(9+nu-6+i);
      r(i)=std::max(0.,lower(i)+.015-x(3+i)+std::pow(std::min(v,0.),2)/(2*acceleration(i)));
      r(i+6)=std::max(0.,x(3+i)-upper(i)+.015+std::pow(std::max(v,0.),2)/(2*acceleration(i)));
    }
    return r;
  }
  scalar_t getValue(scalar_t,const vector_t& x,const TargetTrajectories&,const PreComputation&)const override{return 5000*residual(x).squaredNorm();}
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t,const vector_t& x,const TargetTrajectories&,const PreComputation&)const override {
    auto r=residual(x);matrix_t J=matrix_t::Zero(12,x.size());
    for(int i=3;i<x.size();++i){auto a=x,b=x;a(i)+=1e-6;b(i)-=1e-6;J.col(i)=(residual(a)-residual(b))/2e-6;}
    ScalarFunctionQuadraticApproximation out;out.setZero(x.size());out.f=5000*r.squaredNorm();out.dfdx=10000*J.transpose()*r;out.dfdxx=10000*J.transpose()*J;return out;
  }
};

struct NativeHeadingCost final : StateCost {
  double heading,weight;
  NativeHeadingCost(double h,double w):heading(h),weight(w){}
  NativeHeadingCost* clone()const override{return new NativeHeadingCost(*this);}
  scalar_t getValue(scalar_t,const vector_t& x,const TargetTrajectories&,const PreComputation&)const override{
    return .5*weight*std::pow(x(2)-heading,2);
  }
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t,const vector_t& x,const TargetTrajectories&,const PreComputation&)const override{
    ScalarFunctionQuadraticApproximation out;out.setZero(x.size());
    out.f=.5*weight*std::pow(x(2)-heading,2);
    out.dfdx(2)=weight*(x(2)-heading);out.dfdxx(2,2)=weight;return out;
  }
};

// Only active for the explicit post-task endpoint continuation. The rate
// state remains the last applied command, not a claim about measured velocity.
struct TerminalHoldVelocityCost final : StateCost {
  double end;int nu;vector_t weights;
  TerminalHoldVelocityCost(double end_,double weight_,int nu_):end(end_),nu(nu_),weights(vector_t::Constant(nu_,weight_)){
  }
  TerminalHoldVelocityCost* clone()const override{return new TerminalHoldVelocityCost(*this);}
  scalar_t getValue(scalar_t t,const vector_t& x,const TargetTrajectories&,const PreComputation&)const override {
    return t<end ? 0. : .5*x.tail(nu).dot(weights.cwiseProduct(x.tail(nu)));
  }
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t t,const vector_t& x,const TargetTrajectories& target,const PreComputation& pre)const override {
    ScalarFunctionQuadraticApproximation out;out.setZero(x.size());
    if(t>=end){out.f=getValue(t,x,target,pre);out.dfdx.tail(nu)=weights.cwiseProduct(x.tail(nu));out.dfdxx.bottomRightCorner(nu,nu)=weights.asDiagonal();}
    return out;
  }
};

// During an explicit physical brake, SLQ predicts the same bounded
// command deceleration. Once the applied body command is zero, its rate and
// acceleration remain exactly zero and the measured base pose is fixed.
struct AppliedCommandBrake final : StateInputConstraint {
  double stamp;vector_t initial,limits;int nu;
  AppliedCommandBrake(double t,const vector_t& velocity,const vector_t& acceleration,int nu_)
      :StateInputConstraint(ConstraintOrder::Linear),stamp(t),initial(velocity),limits(acceleration),nu(nu_){}
  AppliedCommandBrake* clone()const override{return new AppliedCommandBrake(*this);}
  size_t getNumConstraints(scalar_t)const override{return initial.size();}
  vector_t acceleration(scalar_t t)const {
    vector_t out=vector_t::Zero(initial.size());
    for(int i=0;i<out.size();++i)if(std::abs(initial(i))>1e-8 && t<=stamp+std::abs(initial(i))/limits(i))out(i)=-std::copysign(limits(i),initial(i));
    return out;
  }
  vector_t getValue(scalar_t t,const vector_t&,const vector_t& u,const PreComputation&)const override{return u.head(initial.size())-acceleration(t);}
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t t,const vector_t& x,const vector_t& u,const PreComputation& p)const override {
    VectorFunctionLinearApproximation out;out.setZero(initial.size(),x.size(),u.size());out.f=getValue(t,x,u,p);out.dfdu.leftCols(initial.size()).setIdentity();return out;
  }
};

class MeasuredStateInitializer final:public Initializer {
  jev::AccelerationDynamics dynamics_;
  scalar_array_t times_;vector_array_t inputs_;vector_t limits_;
 public:
  MeasuredStateInitializer(bool omni,scalar_array_t times,vector_array_t inputs,vector_t limits,double lateralYawGain=0.)
      :dynamics_(omni,lateralYawGain),times_(std::move(times)),inputs_(std::move(inputs)),limits_(std::move(limits)){}
  MeasuredStateInitializer* clone()const override{return new MeasuredStateInitializer(*this);}
  void compute(scalar_t time,const vector_t& state,scalar_t nextTime,vector_t& input,vector_t& nextState)override {
    input=vector_t::Zero(limits_.size());
    if(times_.size()>=2 && time>=times_.front() && time<=times_.back()) {
      auto upper=std::upper_bound(times_.begin(),times_.end(),time);
      std::size_t i=std::min<std::size_t>(std::max<std::ptrdiff_t>(1,upper-times_.begin())-1,times_.size()-2);
      double fraction=(time-times_[i])/(times_[i+1]-times_[i]);
      input=((1.-fraction)*inputs_[i]+fraction*inputs_[i+1]).cwiseMax(-limits_).cwiseMin(limits_);
    }
    PreComputation pre;double dt=nextTime-time;
    auto a=dynamics_.computeFlowMap(time,state,input,pre);
    auto b=dynamics_.computeFlowMap(time+dt/2,state+dt*a/2,input,pre);
    auto c=dynamics_.computeFlowMap(time+dt/2,state+dt*b/2,input,pre);
    auto d=dynamics_.computeFlowMap(nextTime,state+dt*c,input,pre);
    nextState=state+dt*(a+2*b+2*c+d)/6;
  }
};

std::string solve(const Tree& spec,const Tree& model,const Tree& request) {
  static vector_array_t collisionSeeds;
  static std::string seedPath;
  static double seedStamp=-1e9;
  static scalar_array_t acceptedTimes;
  static vector_array_t acceptedInputs;
  static std::string acceptedPath;
  static double acceptedStamp=-1e9;
  const bool omni=spec.get<std::string>("base_model")=="omnidirectional";
  const int nu=omni?9:8,nx=9+nu;
  double stamp=request.get<double>("state_stamp_s"),horizon=request.get<double>("horizon_s",1.);
  vector_t state=array(request.get_child("state"));
  if(state.size()!=nx || !std::isfinite(stamp) || stamp<0 || !std::isfinite(horizon) || horizon<=.01 || horizon>2.) throw std::runtime_error("invalid measured state/horizon");
  if(request.get<std::string>("rate_state_semantics","")!="last_applied_velocity_command")throw std::runtime_error("input-rate state must identify the last applied velocity command");
  CameraModel camera(model);Pose measured=matrix(request.get_child("T_world_base"));
  camera.height=measured(2,3);camera.measuredTilt=Eigen::AngleAxisd(-state(2),Eigen::Vector3d::UnitZ()).toRotationMatrix()*measured.topLeftCorner<3,3>();
  if(request.get<std::string>("frame")!="world_optical")throw std::runtime_error("explicit world optical reference required");
  auto times=array(request.get_child("absolute_times_s"));scalar_array_t timestamps(times.data(),times.data()+times.size());vector_array_t poses;
  for(const auto& p:request.get_child("camera_poses"))poses.push_back(packed(matrix(p.second)));
  if(times.size()<2 || poses.size()!=timestamps.size() || timestamps.front()>stamp || timestamps.back()<stamp+horizon-1e-9)throw std::runtime_error("reference does not cover the prediction interval");
  for(int i=1;i<times.size();++i)if(times(i)<=times(i-1))throw std::runtime_error("reference time must increase");
  const bool terminalHold=bool(request.get_child_optional("terminal_hold"));
  double taskEnd=0.;
  if(terminalHold){
    const auto& hold=request.get_child("terminal_hold");
    if(omni || hold.get<std::string>("mode")!="constant_original_endpoint_after_end")throw std::runtime_error("unsupported terminal continuation");
    taskEnd=hold.get<double>("original_end_s");
    const auto endpoint=packed(matrix(hold.get_child("T_world_original_endpoint")));
    if(!std::isfinite(taskEnd))throw std::runtime_error("invalid original task end");
    for(int i=0;i<times.size();++i)if(times(i)>=taskEnd-1e-9 &&
        (poses[i].head(3)-endpoint.head(3)).norm()+std::min((poses[i].tail(4)-endpoint.tail(4)).norm(),(poses[i].tail(4)+endpoint.tail(4)).norm())>1e-8)
      throw std::runtime_error("terminal prediction changed the original endpoint");
  }
  TargetTrajectories target(std::move(timestamps),std::move(poses));
  vector_t speed=vector_t::Constant(nu,.65),acceleration=vector_t::Ones(nu);
  speed(0)=.22;acceleration(0)=.45;
  speed(omni?2:1)=.35;acceleration(omni?2:1)=.65;
  if(omni){speed(1)=.22;acceleration(1)=.45;}
  OptimalControlProblem problem;
  const double lateralYawGain=request.get<double>("measured_nonholonomic_lateral_yaw_gain_m",0.);
  problem.dynamicsPtr=std::make_unique<jev::AccelerationDynamics>(omni,lateralYawGain);
  problem.stateCostPtr->add("world_optical_camera",std::make_unique<CameraCost>(camera,10.));
  problem.finalCostPtr->add("terminal_world_optical_camera",std::make_unique<CameraCost>(camera,50.));
  if(auto heading=request.get_optional<double>("native_base_heading_target_rad")){
    if(omni || !std::isfinite(*heading))throw std::runtime_error("invalid native nonholonomic allocation");
    // Preserve camera hold while preparing the differential base orientation.
    double weight=request.get<bool>("native_allocation_preparation",false)?5.:.2;
    problem.stateCostPtr->add("native_camera_tangent_heading",std::make_unique<NativeHeadingCost>(*heading,weight));
    problem.finalCostPtr->add("native_camera_tangent_heading_terminal",std::make_unique<NativeHeadingCost>(*heading,5.*weight));
  }
  if(terminalHold){
    problem.stateCostPtr->add("post_task_command_stop",std::make_unique<TerminalHoldVelocityCost>(taskEnd,10.,nu));
    problem.finalCostPtr->add("post_task_terminal_command_stop",std::make_unique<TerminalHoldVelocityCost>(taskEnd,50.,nu));
  }
  const bool chassisBrake=terminalHold && request.get<bool>("terminal_hold.chassis_brake_applied",false);
  const bool allBrake=terminalHold && request.get<bool>("terminal_hold.all_actuators_brake_applied",false);
  if(allBrake && (!chassisBrake || stamp<taskEnd+4.-1e-9))throw std::runtime_error("all-actuator brake precedes its declared last-one-second stop window");
  if(chassisBrake){
    if(stamp<taskEnd-1e-9)throw std::runtime_error("chassis stop cannot replace the active task");
    problem.equalityConstraintPtr->add("applied_physical_chassis_brake",std::make_unique<AppliedCommandBrake>(stamp,state.segment(9,allBrake?nu:nu-6),acceleration.head(allBrake?nu:nu-6),nu));
  }
  if(model.get_child_optional("collision_shapes")) {
    if(request.get_child_optional("environment")) {
      vector_array_t anchors{state};
      if(seedPath==request.get<std::string>("camera_path_id","") && stamp>=seedStamp && stamp-seedStamp<=.25)
        anchors.insert(anchors.end(),collisionSeeds.begin(),collisionSeeds.end());
      problem.stateCostPtr->add("actual_collider_box_separation",
          std::make_unique<PredictiveLocalCollisionCost>(CollisionCost(camera,model),anchors,20000.));
    } else
      problem.stateCostPtr->add("actual_collider_box_separation",std::make_unique<LocalCollisionCost>(CollisionCost(camera,model),state));
  }
  std::unique_ptr<ObservedEnvironment> environment;
  if(request.get_child_optional("environment")) {
    environment=std::make_unique<ObservedEnvironment>(camera,model,request.get_child("environment"),stamp);
    problem.stateCostPtr->add("observed_environment_occupancy",std::make_unique<LocalEnvironmentCost>(*environment,stamp,state));
  }
  problem.stateCostPtr->add("anticipatory_joint_braking",std::make_unique<BrakingCost>(camera,acceleration));
  problem.costPtr->add("acceleration_effort",std::make_unique<InputCost>(nu));
  vector_t h(12+4*nu);matrix_t C=matrix_t::Zero(h.size(),nx),D=matrix_t::Zero(h.size(),nu);
  h.head(6)=-camera.lower;h.segment(6,6)=camera.upper;
  C.block(0,3,6,6).setIdentity();C.block(6,3,6,6)=-matrix_t::Identity(6,6);
  h.segment(12,nu)=.95*speed;h.segment(12+nu,nu)=.95*speed;
  C.block(12,9,nu,nu).setIdentity();C.block(12+nu,9,nu,nu)=-matrix_t::Identity(nu,nu);
  h.segment(12+2*nu,nu)=(environment ? .85 : .95)*acceleration;h.tail(nu)=(environment ? .85 : .95)*acceleration;
  D.block(12+2*nu,0,nu,nu).setIdentity();D.bottomRows(nu)=-matrix_t::Identity(nu,nu);
  problem.softConstraintPtr->add("physical_model_bounds",std::make_unique<StateInputSoftConstraint>(
      std::make_unique<LinearStateInputConstraint>(h,C,D),std::make_unique<RelaxedBarrierPenalty>(RelaxedBarrierPenalty::Config(environment ? 2. : .2,environment ? .001 : .005))));
  // Identity mode events make SLQ's controller/rollout represent the known
  // acceleration discontinuities exactly, without resetting any state.
  ModeSchedule brakeSchedule;
  if(chassisBrake){
    scalar_array_t events;
    for(int i=0;i<(allBrake?nu:nu-6);++i){double stop=stamp+std::abs(state(9+i))/acceleration(i);if(stop>stamp+1e-8 && stop<stamp+horizon-1e-8)events.push_back(stop);}
    std::sort(events.begin(),events.end());
    events.erase(std::unique(events.begin(),events.end(),[](double a,double b){return std::abs(a-b)<1e-8;}),events.end());
    brakeSchedule=ModeSchedule(events,size_array_t(events.size()+1,0));
  }
  rollout::Settings rs;rs.timeStep=.01;rs.absTolODE=1e-7;rs.relTolODE=1e-5;
  TimeTriggeredRollout rollout(*problem.dynamicsPtr,rs);
  bool warm=environment && !chassisBrake && acceptedPath==request.get<std::string>("camera_path_id","")
            && stamp>=acceptedStamp && stamp-acceptedStamp<=.25;
  std::unique_ptr<Initializer> initializer;
  if(environment)initializer=std::make_unique<MeasuredStateInitializer>(omni,warm?acceptedTimes:scalar_array_t{},
      warm?acceptedInputs:vector_array_t{},.85*acceleration,lateralYawGain);
  else initializer=std::make_unique<DefaultInitializer>(nu);
  mpc::Settings ms;ms.timeHorizon_=horizon;ms.coldStart_=true;
  ddp::Settings ds;ds.algorithm_=ddp::Algorithm::SLQ;ds.nThreads_=1;ds.maxNumIterations_=5;ds.displayInfo_=false;
  GaussNewtonDDP_MPC mpc(ms,ds,rollout,problem,*initializer);
  mpc.getSolverPtr()->setReferenceManager(std::make_shared<ReferenceManager>(target,brakeSchedule));
  const auto start=std::chrono::steady_clock::now();bool solved=mpc.run(stamp,state);
  PrimalSolution policy;mpc.getSolverPtr()->getPrimalSolution(stamp+horizon,&policy);
  if(!solved || policy.stateTrajectory_.empty())throw std::runtime_error("native SLQ returned no policy");
  collisionSeeds.clear();
  for(double fraction:{.4,.8,1.}) {
    auto found=std::lower_bound(policy.timeTrajectory_.begin(),policy.timeTrajectory_.end(),stamp+fraction*horizon);
    std::size_t index=std::min<std::size_t>(found-policy.timeTrajectory_.begin(),policy.stateTrajectory_.size()-1);
    collisionSeeds.push_back(policy.stateTrajectory_[index]);
  }
  seedPath=request.get<std::string>("camera_path_id","");seedStamp=stamp;
  double jointMargin=1e9,maxVelocity=0.,maxAcceleration=0.;
  for(const auto& x:policy.stateTrajectory_){if(!x.allFinite())throw std::runtime_error("nonfinite native state");jointMargin=std::min(jointMargin,std::min((x.segment(3,6)-camera.lower).minCoeff(),(camera.upper-x.segment(3,6)).minCoeff()));maxVelocity=std::max(maxVelocity,x.tail(nu).cwiseAbs().cwiseQuotient(speed).maxCoeff());}
  for(const auto& a:policy.inputTrajectory_){if(!a.allFinite())throw std::runtime_error("nonfinite native acceleration");maxAcceleration=std::max(maxAcceleration,a.cwiseAbs().cwiseQuotient(acceleration).maxCoeff());}
  double collisionMargin=1e9;
  if(model.get_child_optional("collision_shapes")){
    CollisionCost check(camera,model);
    for(const auto& x:policy.stateTrajectory_)collisionMargin=std::min(collisionMargin,.025-check.residual(x,false).first.maxCoeff());
  }
  if(collisionMargin<.005)throw std::runtime_error("predicted conservative convex-hull self-collision margin rejected");
  if(environment) {
    double minimum=1e9;
    for(std::size_t k=0;k<policy.stateTrajectory_.size();++k) {
      auto distances=environment->evaluate(policy.timeTrajectory_[k],policy.stateTrajectory_[k],false).distance;
      if(distances.size())minimum=std::min(minimum,distances.minCoeff());
    }
    if(minimum<.025-1e-6)throw std::runtime_error("predicted measured environment collision margin rejected: "+std::to_string(minimum));
  }

  if(maxVelocity>1.00001 || maxAcceleration>1.00001){std::ostringstream reason;reason<<std::setprecision(12)<<"predicted velocity/acceleration exceeds original execution bound: velocity_ratio="<<maxVelocity<<", acceleration_ratio="<<maxAcceleration;throw std::runtime_error(reason.str());}
  if(jointMargin<0.)throw std::runtime_error("predicted physical joint limit violation");
  acceptedTimes=policy.timeTrajectory_;acceptedInputs=policy.inputTrajectory_;
  acceptedPath=request.get<std::string>("camera_path_id","");acceptedStamp=stamp;
  std::ostringstream out;out<<std::setprecision(17)<<"{\"ok\":true,\"framework\":\"OCS2_SLQ_input_rate\",\"rate_state_semantics\":\"last_applied_velocity_command\",\"chassis_brake_mode\":\""<<(chassisBrake?(allBrake?(state.tail(nu).norm()<1e-12?"fixed_measured_configuration":"all_actuator_constrained_deceleration"):(state.segment(9,nu-6).norm()<1e-12?"fixed_measured_base":"constrained_deceleration")):"inactive")<<"\",\"request_id\":"<<request.get<unsigned long>("request_id")<<",\"state_stamp_s\":"<<stamp<<",\"solver_wall_s\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"iterations\":"<<mpc.getSolverPtr()->getNumIterations()<<",\"maximum_velocity_ratio\":"<<maxVelocity<<",\"maximum_acceleration_ratio\":"<<maxAcceleration<<",\"minimum_self_clearance_lower_bound_m\":"<<collisionMargin<<",\"minimum_joint_margin_rad\":"<<jointMargin<<",\"times_s\":[";
  for(std::size_t i=0;i<policy.timeTrajectory_.size();++i){if(i)out<<',';out<<policy.timeTrajectory_[i];}
  auto rows=[&](const char* key,const vector_array_t& values){out<<"],\""<<key<<"\":[";for(std::size_t i=0;i<values.size();++i){if(i)out<<',';out<<'[';for(int j=0;j<values[i].size();++j){if(j)out<<',';out<<values[i](j);}out<<']';}};
  rows("states",policy.stateTrajectory_);rows("accelerations",policy.inputTrajectory_);out<<"]}";return out.str();
}

int main(int argc,char** argv)try {
  if(argc!=3)throw std::runtime_error("usage: camera_mpc_worker backend.json model.json");
  Tree spec,model;boost::property_tree::read_json(argv[1],spec);boost::property_tree::read_json(argv[2],model);
  std::string line;
  while(std::getline(std::cin,line)) {
    try{Tree request;std::istringstream input(line);boost::property_tree::read_json(input,request);std::cout<<solve(spec,model,request)<<std::endl;}
    catch(const std::exception& e){Tree error;error.put("reason",e.what());std::ostringstream output;boost::property_tree::write_json(output,error,false);auto detail=output.str();if(!detail.empty() && detail.back()=='\n')detail.pop_back();std::cout<<"{\"ok\":false,\"detail\":"<<detail<<"}"<<std::endl;}
  }
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
