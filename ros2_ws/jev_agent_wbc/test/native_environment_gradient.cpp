// Compare native signed-distance Jacobians to independent central differences.
#include "ObservedEnvironmentCost.h"
#include "CollisionCost.h"
#include "AccelerationDynamics.h"
#include <iostream>
#ifndef JEV_TEST_OMNI
#define JEV_TEST_OMNI 0
#endif
int main(int argc,char** argv) {
  if(argc!=3)return 2;
  Tree model,request;
  boost::property_tree::read_json(argv[1],model);
  boost::property_tree::read_json(argv[2],request);
  CameraModel camera(model);
  auto x=array(request.get_child("state"));
  auto base=matrix(request.get_child("T_world_base"));
  camera.height=base(2,3);
  camera.measuredTilt=Eigen::AngleAxisd(-x(2),Eigen::Vector3d::UnitZ()).toRotationMatrix()*base.topLeftCorner<3,3>();
  double stamp=request.get<double>("state_stamp_s"),time=stamp+.17;
  ObservedEnvironment geometry(camera,model,request.get_child("environment"),stamp);
  auto result=geometry.evaluate(time,x);
  double maximum=0.;
  for(int j=0;j<x.size();++j) {
    auto a=x,b=x;a(j)+=1e-6;b(j)-=1e-6;
    vector_t derivative=(geometry.evaluate(time,a).distance-geometry.evaluate(time,b).distance)/2e-6;
    maximum=std::max(maximum,(derivative-result.gradient.col(j)).cwiseAbs().maxCoeff());
  }
  vector_t timeDerivative=(geometry.evaluate(time+1e-6,x).distance-geometry.evaluate(time-1e-6,x).distance)/2e-6;
  double timeError=(timeDerivative-result.timeDerivative).cwiseAbs().maxCoeff();
  auto kinematics=camera.differential(x);
  Pose goal=kinematics.first;goal.topLeftCorner<3,3>()=Eigen::AngleAxisd(.7,Eigen::Vector3d(1.,2.,3.).normalized()).toRotationMatrix()*goal.topLeftCorner<3,3>();
  auto e=error(kinematics.first,goal);auto W=skew(e.tail<3>());double theta=e.tail<3>().norm();
  matrix_t jac=kinematics.second;
  double c=(1.-.5*theta/std::tan(.5*theta))/(theta*theta);
  jac.bottomRows(3)=(Eigen::Matrix3d::Identity()-.5*W+c*W*W)*goal.topLeftCorner<3,3>().transpose()*kinematics.second.bottomRows(3);
  double cameraError=0.;
  for(int j=0;j<x.size();++j) {
    auto a=x,b=x;a(j)+=1e-6;b(j)-=1e-6;
    vector_t numerical=(error(camera.camera(a),goal)-error(camera.camera(b),goal))/2e-6;
    cameraError=std::max(cameraError,(numerical-jac.col(j)).cwiseAbs().maxCoeff());
  }
  // Force an active observed-distance residual and compare the sparse
  // accumulation with the independent dense Gauss--Newton expression.
  Tree activeScene=request.get_child("environment");
  auto& spheres=activeScene.get_child("spheres");
  if(spheres.empty())return 2;
  double firstGap=1e9;
  for(int i=0;i<result.distance.size();i+=spheres.size())
    firstGap=std::min(firstGap,result.distance(i));
  auto& first=spheres.begin()->second;
  first.put("radius_m",first.get<double>("radius_m")+firstGap+.035);
  ObservedEnvironment active(camera,model,activeScene,stamp);
  active.retainCostNeighborhood(time,x);
  auto dense=active.evaluate(time,x);
  constexpr double softMargin=JEV_TEST_OMNI ? .14 : .065;
  vector_t residual=(vector_t::Constant(dense.distance.size(),softMargin)-dense.distance).cwiseMax(0.);
  matrix_t J=-dense.gradient;
  for(int i=0;i<residual.size();++i)if(residual(i)==0.)J.row(i).setZero();
  LocalEnvironmentCost cost(active,time,x);
  PreComputation pre;TargetTrajectories target;
  auto quad=cost.getQuadraticApproximation(time,x,target,pre);
  double quadraticError=std::max({std::abs(quad.f-20000*residual.squaredNorm()),
    (quad.dfdx-40000*J.transpose()*residual).cwiseAbs().maxCoeff(),
    (quad.dfdxx-40000*J.transpose()*J).cwiseAbs().maxCoeff()});
  CollisionCost exactSelf(camera,model);LocalCollisionCost localSelf(exactSelf,x);
  double selfAnchorError=std::abs(exactSelf.getValue(time,x,target,pre)-
                                localSelf.getValue(time,x,target,pre));
#if JEV_TEST_OMNI
  jev::AccelerationDynamics dynamics(true);
  vector_t dx=vector_t::Zero(18),u=vector_t::Zero(9);
#else
  jev::AccelerationDynamics dynamics(false,-.27);
  vector_t dx=vector_t::Zero(17),u=vector_t::Zero(8);
#endif
  dx(2)=.7;dx(9)=.13;dx(10)=.22;
  auto linear=dynamics.linearApproximation(time,dx,u,pre);
  double dynamicsError=0.;
  for(int j=0;j<dx.size();++j){
    vector_t a=dx,b=dx;a(j)+=1e-6;b(j)-=1e-6;
    vector_t numerical=(dynamics.computeFlowMap(time,a,u,pre)-dynamics.computeFlowMap(time,b,u,pre))/2e-6;
    dynamicsError=std::max(dynamicsError,(numerical-linear.dfdx.col(j)).cwiseAbs().maxCoeff());
  }
  bool passed=maximum<1e-6 && timeError<1e-6 && cameraError<1e-6 &&
              quadraticError<1e-7 && selfAnchorError<1e-7 && dynamicsError<1e-7 && residual.maxCoeff()>0.;
  std::cout<<"{\"state_gradient_max_error\":"<<maximum<<",\"time_gradient_max_error\":"<<timeError
           <<",\"camera_error_gradient_max_error\":"<<cameraError
           <<",\"sparse_quadratic_max_error\":"<<quadraticError
           <<",\"self_collision_anchor_cost_error\":"<<selfAnchorError
           <<",\"measured_origin_dynamics_gradient_error\":"<<dynamicsError
           <<",\"active_environment_residual_m\":"<<residual.maxCoeff()
           <<",\"rows\":"<<result.distance.size()<<",\"passed\":"<<(passed?"true":"false")<<"}\n";
  return passed?0:1;
}
