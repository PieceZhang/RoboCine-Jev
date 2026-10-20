// Persistent genuine OCS2 SLQ worker. One JSON request/response per line.
// The physical supervisor owns command expiry, hard limits and collision brakes.
#include "CameraMpcModel.h"
#include "AccelerationDynamics.h"
#include "CollisionCost.h"
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

std::string solve(const Tree& spec,const Tree& model,const Tree& request) {
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
  TargetTrajectories target(std::move(timestamps),std::move(poses));
  vector_t speed=vector_t::Constant(nu,.65),acceleration=vector_t::Ones(nu);
  speed(0)=.22;acceleration(0)=.45;
  speed(omni?2:1)=.35;acceleration(omni?2:1)=.65;
  if(omni){speed(1)=.22;acceleration(1)=.45;}
  OptimalControlProblem problem;
  problem.dynamicsPtr=std::make_unique<jev::AccelerationDynamics>(omni);
  problem.stateCostPtr->add("world_optical_camera",std::make_unique<CameraCost>(camera,10.));
  problem.finalCostPtr->add("terminal_world_optical_camera",std::make_unique<CameraCost>(camera,50.));
  if(model.get_child_optional("collision_shapes"))problem.stateCostPtr->add("actual_collider_box_separation",std::make_unique<LocalCollisionCost>(CollisionCost(camera,model),state));
  problem.stateCostPtr->add("anticipatory_joint_braking",std::make_unique<BrakingCost>(camera,acceleration));
  problem.costPtr->add("acceleration_effort",std::make_unique<InputCost>(nu));
  vector_t h(12+4*nu);matrix_t C=matrix_t::Zero(h.size(),nx),D=matrix_t::Zero(h.size(),nu);
  h.head(6)=-camera.lower;h.segment(6,6)=camera.upper;
  C.block(0,3,6,6).setIdentity();C.block(6,3,6,6)=-matrix_t::Identity(6,6);
  h.segment(12,nu)=.95*speed;h.segment(12+nu,nu)=.95*speed;
  C.block(12,9,nu,nu).setIdentity();C.block(12+nu,9,nu,nu)=-matrix_t::Identity(nu,nu);
  h.segment(12+2*nu,nu)=.95*acceleration;h.tail(nu)=.95*acceleration;
  D.block(12+2*nu,0,nu,nu).setIdentity();D.bottomRows(nu)=-matrix_t::Identity(nu,nu);
  problem.softConstraintPtr->add("physical_model_bounds",std::make_unique<StateInputSoftConstraint>(
      std::make_unique<LinearStateInputConstraint>(h,C,D),std::make_unique<RelaxedBarrierPenalty>(RelaxedBarrierPenalty::Config(.2,.005))));
  rollout::Settings rs;rs.timeStep=.01;rs.absTolODE=1e-7;rs.relTolODE=1e-5;
  TimeTriggeredRollout rollout(*problem.dynamicsPtr,rs);DefaultInitializer initializer(nu);
  mpc::Settings ms;ms.timeHorizon_=horizon;ms.coldStart_=true;
  ddp::Settings ds;ds.algorithm_=ddp::Algorithm::SLQ;ds.nThreads_=1;ds.maxNumIterations_=5;ds.displayInfo_=false;
  GaussNewtonDDP_MPC mpc(ms,ds,rollout,problem,initializer);
  mpc.getSolverPtr()->setReferenceManager(std::make_shared<ReferenceManager>(target));
  const auto start=std::chrono::steady_clock::now();bool solved=mpc.run(stamp,state);
  PrimalSolution policy;mpc.getSolverPtr()->getPrimalSolution(stamp+horizon,&policy);
  if(!solved || policy.stateTrajectory_.empty())throw std::runtime_error("native SLQ returned no policy");
  double jointMargin=1e9,maxVelocity=0.,maxAcceleration=0.;
  for(const auto& x:policy.stateTrajectory_){if(!x.allFinite())throw std::runtime_error("nonfinite native state");jointMargin=std::min(jointMargin,std::min((x.segment(3,6)-camera.lower).minCoeff(),(camera.upper-x.segment(3,6)).minCoeff()));maxVelocity=std::max(maxVelocity,x.tail(nu).cwiseAbs().cwiseQuotient(speed).maxCoeff());}
  for(const auto& a:policy.inputTrajectory_){if(!a.allFinite())throw std::runtime_error("nonfinite native acceleration");maxAcceleration=std::max(maxAcceleration,a.cwiseAbs().cwiseQuotient(acceleration).maxCoeff());}
  double collisionMargin=1e9;
  if(model.get_child_optional("collision_shapes")){
    CollisionCost check(camera,model);
    for(const auto& x:policy.stateTrajectory_)collisionMargin=std::min(collisionMargin,.025-check.residual(x,false).first.maxCoeff());
  }
  if(collisionMargin<.005)throw std::runtime_error("predicted conservative convex-hull self-collision margin rejected");
  if(maxVelocity>1.00001 || maxAcceleration>1.00001){std::ostringstream reason;reason<<std::setprecision(12)<<"predicted velocity/acceleration exceeds original execution bound: velocity_ratio="<<maxVelocity<<", acceleration_ratio="<<maxAcceleration;throw std::runtime_error(reason.str());}
  if(jointMargin<0.)throw std::runtime_error("predicted physical joint limit violation");
  std::ostringstream out;out<<std::setprecision(17)<<"{\"ok\":true,\"framework\":\"OCS2_SLQ_input_rate\",\"rate_state_semantics\":\"last_applied_velocity_command\",\"request_id\":"<<request.get<unsigned long>("request_id")<<",\"state_stamp_s\":"<<stamp<<",\"solver_wall_s\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"iterations\":"<<mpc.getSolverPtr()->getNumIterations()<<",\"maximum_velocity_ratio\":"<<maxVelocity<<",\"maximum_acceleration_ratio\":"<<maxAcceleration<<",\"minimum_self_clearance_lower_bound_m\":"<<collisionMargin<<",\"minimum_joint_margin_rad\":"<<jointMargin<<",\"times_s\":[";
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
