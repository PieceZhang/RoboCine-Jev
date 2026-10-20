#include "CameraMpcModel.h"

int main(int argc,char** argv) try {
  if(argc<3 || argc>5) throw std::runtime_error("usage: camera_mpc_probe backend.json model.json [fixed_world_camera_request.json [policy.json]]");
  Tree spec,tree;boost::property_tree::read_json(argv[1],spec);boost::property_tree::read_json(argv[2],tree);
  const bool omni=spec.get<std::string>("base_model")=="omnidirectional";const int nu=omni ? 9 : 8;
  CameraModel camera(tree);vector_t x=vector_t::Zero(9);x.tail(6)=array(spec.get_child("initial_arm"));
  Pose start=camera.camera(x),goal=start;
  goal.topRightCorner<3,1>()+=Eigen::Vector3d(.06,.08,.04);
  goal.topLeftCorner<3,3>()=Eigen::AngleAxisd(.06,Eigen::Vector3d(.4,.5,-.6).normalized()).toRotationMatrix()*start.topLeftCorner<3,3>();
  double positionTolerance=.02,orientationTolerance=.05;
  TargetTrajectories target({0.,1.},{packed(start),packed(goal)});
  if(argc>=4) {
    Tree request;boost::property_tree::read_json(argv[3],request);
    positionTolerance=request.get<double>("position_tolerance_m",.02);orientationTolerance=request.get<double>("orientation_tolerance_rad",.05);
    if(!std::isfinite(positionTolerance) || !std::isfinite(orientationTolerance) || positionTolerance<=0 || orientationTolerance<=0) throw std::runtime_error("positive finite tolerances required");
    if(request.get<std::string>("frame")!="world") throw std::runtime_error("native solver accepts explicit fixed world references only");
    x=array(request.get_child("initial_state"));if(x.size()!=9) throw std::runtime_error("expected nine internal measured state coordinates");
    auto times=array(request.get_child("timestamps_s"));scalar_array_t timestamps(times.data(),times.data()+times.size());vector_array_t poses;
    for(const auto& item:request.get_child("camera_poses")) poses.push_back(packed(matrix(item.second)));
    if(timestamps.size()<2 || timestamps.size()!=poses.size() || timestamps.front()!=0. || timestamps.back()<=0. || timestamps.back()>10.) throw std::runtime_error("invalid reference horizon");
    for(std::size_t i=1;i<timestamps.size();++i) if(timestamps[i]<=timestamps[i-1]) throw std::runtime_error("timestamps must increase strictly");
    target=TargetTrajectories(std::move(timestamps),std::move(poses));goal=reference(target.timeTrajectory.back(),target);
  }
  const double horizon=target.timeTrajectory.back();
  OptimalControlProblem problem;
  problem.dynamicsPtr=std::make_unique<jev::WheeledDynamics>(omni);
  problem.stateCostPtr->add("optical_camera_6d",std::make_unique<CameraCost>(camera,1.));
  problem.finalCostPtr->add("terminal_optical_camera_6d",std::make_unique<CameraCost>(camera,5.));
  problem.costPtr->add("actuator_regularization",std::make_unique<InputCost>(nu));
  vector_t speed=vector_t::Constant(nu,.65);speed(0)=.22;speed(omni ? 2 : 1)=.35;if(omni) speed(1)=.22;
  vector_t constant(12+2*nu);matrix_t C=matrix_t::Zero(constant.size(),9),D=matrix_t::Zero(constant.size(),nu);
  constant.head(6)=-camera.lower;constant.segment(6,6)=camera.upper;
  C.block(0,3,6,6).setIdentity();C.block(6,3,6,6)=-matrix_t::Identity(6,6);
  constant.segment(12,nu)=speed;constant.tail(nu)=speed;
  D.block(12,0,nu,nu).setIdentity();D.bottomRows(nu)=-matrix_t::Identity(nu,nu);
  problem.softConstraintPtr->add("joint_and_velocity_barriers",std::make_unique<StateInputSoftConstraint>(
      std::make_unique<LinearStateInputConstraint>(constant,C,D),std::make_unique<RelaxedBarrierPenalty>(RelaxedBarrierPenalty::Config(.001,.005))));
  rollout::Settings rs;rs.timeStep=.01;rs.absTolODE=1e-8;rs.relTolODE=1e-6;
  TimeTriggeredRollout rollout(*problem.dynamicsPtr,rs);DefaultInitializer initializer(nu);
  mpc::Settings ms;ms.timeHorizon_=horizon;ms.coldStart_=true;
  ddp::Settings ds;ds.algorithm_=ddp::Algorithm::SLQ;ds.nThreads_=1;ds.maxNumIterations_=12;ds.displayInfo_=false;
  GaussNewtonDDP_MPC mpc(ms,ds,rollout,problem,initializer);
  auto manager=std::make_shared<ReferenceManager>(target);mpc.getSolverPtr()->setReferenceManager(manager);
  const auto clock=std::chrono::steady_clock::now();bool solved=mpc.run(0.,x);
  PrimalSolution policy;mpc.getSolverPtr()->getPrimalSolution(horizon,&policy);
  double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-clock).count();
  if(policy.stateTrajectory_.empty() || policy.inputTrajectory_.empty()) throw std::runtime_error("OCS2 returned no policy");
  auto e=error(camera.camera(policy.stateTrajectory_.back()),goal);double maxSpeedRatio=0.,margin=1e9;
  for(const auto& u:policy.inputTrajectory_) maxSpeedRatio=std::max(maxSpeedRatio,u.cwiseAbs().cwiseQuotient(speed).maxCoeff());
  for(const auto& state:policy.stateTrajectory_) margin=std::min(margin,std::min((state.tail(6)-camera.lower).minCoeff(),(camera.upper-state.tail(6)).minCoeff()));
  double positionSquared=0.,orientationSquared=0.,maxPositionError=0.,maxOrientationError=0.;bool finite=true;
  for(std::size_t i=0;i<policy.stateTrajectory_.size();++i) {
    finite=finite && policy.stateTrajectory_[i].allFinite();
    auto residual=error(camera.camera(policy.stateTrajectory_[i]),reference(policy.timeTrajectory_[i],target));
    positionSquared+=residual.head<3>().squaredNorm();orientationSquared+=residual.tail<3>().squaredNorm();
    maxPositionError=std::max(maxPositionError,residual.head<3>().norm());maxOrientationError=std::max(maxOrientationError,residual.tail<3>().norm());
  }
  for(const auto& u:policy.inputTrajectory_) finite=finite && u.allFinite();
  auto delta=policy.stateTrajectory_.back()-x;bool passed=finite &&solved && maxPositionError<=positionTolerance && maxOrientationError<=orientationTolerance && maxSpeedRatio<=1. && margin>=0.;
  std::cout << "{\"framework\":\"OCS2 GaussNewtonDDP_MPC/SLQ\",\"scope\":\"one model prediction horizon, no physics or environment constraints\","
            << "\"state_dim\":9,\"input_dim\":"<<nu<<",\"policy_samples\":"<<policy.stateTrajectory_.size()
            <<",\"horizon_s\":"<<horizon<<",\"rollout_dt_s\":0.01,\"max_iterations\":12,\"iterations\":"<<mpc.getSolverPtr()->getNumIterations()
            <<",\"cpu_only\":true,\"camera_position_rmse_m\":"<<std::sqrt(positionSquared/policy.stateTrajectory_.size())
            <<",\"camera_orientation_rmse_rad\":"<<std::sqrt(orientationSquared/policy.stateTrajectory_.size())
            <<",\"rmse_sampling\":\"unweighted_native_policy_knots\",\"maximum_position_error_m\":"<<maxPositionError
            <<",\"maximum_orientation_error_rad\":"<<maxOrientationError<<",\"position_tolerance_m\":"<<positionTolerance
            <<",\"orientation_tolerance_rad\":"<<orientationTolerance<<",\"solver_wall_s\":"<<wall<<",\"terminal_position_error_m\":"<<e.head<3>().norm()
            <<",\"terminal_orientation_error_rad\":"<<e.tail<3>().norm()<<",\"predicted_base_delta_m\":"<<delta.head<2>().norm()
            <<",\"predicted_arm_delta_rad\":"<<delta.tail(6).norm()<<",\"maximum_speed_ratio\":"<<maxSpeedRatio
            <<",\"minimum_joint_margin_rad\":"<<margin<<",\"mpc_executed\":true,\"physical_execution_verified\":false,\"passed\":"<<(passed?"true":"false")<<"}\n";
  if(argc==5 && passed) {
    std::ofstream output(argv[4]);if(!output) throw std::runtime_error("cannot write policy output");
    output.precision(17);output << "{\"schema\":\"ocs2_wheeled_internal_policy/v1\",\"physical_commands_issued\":false,\"timestamps_s\":[";
    for(std::size_t i=0;i<policy.timeTrajectory_.size();++i) { if(i)output<<',';output<<policy.timeTrajectory_[i]; }
    auto rows=[&](const char* name,const vector_array_t& values) {
      output << "],\""<<name<<"\":[";
      for(std::size_t i=0;i<values.size();++i) { if(i)output<<',';output<<'[';for(int j=0;j<values[i].size();++j){if(j)output<<',';output<<values[i](j);}output<<']'; }
    };
    rows("states",policy.stateTrajectory_);rows("controls",policy.inputTrajectory_);output<<"]}\n";
  }
  return passed?0:2;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 2; }
