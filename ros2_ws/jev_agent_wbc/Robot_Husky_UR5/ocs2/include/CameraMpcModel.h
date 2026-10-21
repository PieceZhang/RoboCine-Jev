#pragma once
// Genuine OCS2 SLQ/MPC with a robot-specific USD-derived optical-frame cost.
// This executable performs one prediction horizon, never a PhysX execution.
#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>
#include <boost/property_tree/json_parser.hpp>
#include <ocs2_core/constraint/LinearStateInputConstraint.h>
#include <ocs2_core/cost/StateCost.h>
#include <ocs2_core/cost/StateInputCost.h>
#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_ddp/GaussNewtonDDP_MPC.h>
#include <ocs2_oc/rollout/TimeTriggeredRollout.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include "WheeledDynamics.h"

using namespace ocs2;
using Tree=boost::property_tree::ptree;
using Pose=Eigen::Matrix4d;

vector_t array(const Tree& tree) {
  vector_t out(tree.size()); int i=0;
  for(const auto& item:tree) out(i++)=item.second.get_value<double>();
  if(!out.allFinite()) throw std::runtime_error("nonfinite input array");
  return out;
}
Pose matrix(const Tree& tree) {
  if(tree.size()!=4) throw std::runtime_error("SE3 must have four rows");
  Pose out; int i=0;
  for(const auto& row:tree) { auto values=array(row.second); if(values.size()!=4) throw std::runtime_error("bad SE3 row"); out.row(i++)=values.transpose(); }
  if(i!=4 || !out.allFinite() || (out.row(3)-Eigen::RowVector4d(0,0,0,1)).norm()>1e-8
      || (out.topLeftCorner<3,3>().transpose()*out.topLeftCorner<3,3>()-Eigen::Matrix3d::Identity()).norm()>1e-7
      || std::abs(out.topLeftCorner<3,3>().determinant()-1)>1e-7) throw std::runtime_error("bad SE3 matrix");
  return out;
}
Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d w;w << 0.,-v.z(),v.y(),v.z(),0.,-v.x(),-v.y(),v.x(),0.;return w;
}

struct CameraModel {
  Pose M, extrinsic;
  Eigen::Matrix3d measuredTilt=Eigen::Matrix3d::Identity();
  double height;
  std::vector<Eigen::Vector3d> omega,velocity;
  vector_t rest,lower,upper;
  explicit CameraModel(const Tree& tree) {
    M=matrix(tree.get_child("T_base_ee_at_authored_pose"));
    extrinsic=matrix(tree.get_child("T_ee_camera_optical"));
    height=matrix(tree.get_child("T_world_base_at_authored_pose"))(2,3);
    rest.resize(6); lower.resize(6);upper.resize(6);int i=0;
    for(const auto& item:tree.get_child("joints")) {
      omega.push_back(array(item.second.get_child("screw_omega")));
      velocity.push_back(array(item.second.get_child("screw_v")));
      rest(i)=item.second.get<double>("q_at_authored_pose");
      lower(i)=item.second.get<double>("lower");upper(i)=item.second.get<double>("upper");++i;
    }
    if(i!=6) throw std::runtime_error("expected six exact USD joints");
  }
  std::pair<Pose,matrix_t> differential(const vector_t& x) const {
    Pose prefix=Pose::Identity();
    prefix.topLeftCorner<3,3>()=Eigen::AngleAxisd(x(2),Eigen::Vector3d::UnitZ()).toRotationMatrix()*measuredTilt;
    prefix.topRightCorner<3,1>() << x(0),x(1),height;
    Eigen::Vector3d basePosition=prefix.topRightCorner<3,1>();
    std::vector<Eigen::Vector3d> wworld(6),vworld(6);
    for(int i=0;i<6;++i) {
      wworld[i]=prefix.topLeftCorner<3,3>()*omega[i];
      vworld[i]=prefix.topLeftCorner<3,3>()*velocity[i]+prefix.topRightCorner<3,1>().cross(wworld[i]);
      double a=x(3+i)-rest(i);auto w=skew(omega[i]);Pose e=Pose::Identity();
      e.topLeftCorner<3,3>()=Eigen::Matrix3d::Identity()+std::sin(a)*w+(1-std::cos(a))*w*w;
      e.topRightCorner<3,1>()=(a*Eigen::Matrix3d::Identity()+(1-std::cos(a))*w+(a-std::sin(a))*w*w)*velocity[i];
      prefix=prefix*e;
    }
    Pose pose=prefix*M*extrinsic;Eigen::Vector3d point=pose.topRightCorner<3,1>();
    matrix_t J=matrix_t::Zero(6,x.size());J(0,0)=J(1,1)=J(5,2)=1.;
    J.block<3,1>(0,2)=Eigen::Vector3d::UnitZ().cross(point-basePosition);
    for(int i=0;i<6;++i){J.block<3,1>(0,3+i)=wworld[i].cross(point)+vworld[i];J.block<3,1>(3,3+i)=wworld[i];}
    return {pose,J};
  }
  Pose camera(const vector_t& x) const {
    Pose base=Pose::Identity();base.topLeftCorner<3,3>()=Eigen::AngleAxisd(x(2),Eigen::Vector3d::UnitZ()).toRotationMatrix()*measuredTilt;
    base.topRightCorner<3,1>() << x(0),x(1),height;
    for(int i=0;i<6;++i) {
      double a=x(3+i)-rest(i);auto w=skew(omega[i]);Pose e=Pose::Identity();
      e.topLeftCorner<3,3>()=Eigen::Matrix3d::Identity()+std::sin(a)*w+(1-std::cos(a))*w*w;
      e.topRightCorner<3,1>()=(a*Eigen::Matrix3d::Identity()+(1-std::cos(a))*w+(a-std::sin(a))*w*w)*velocity[i];
      base=base*e;
    }
    return base*M*extrinsic;
  }
};

Pose reference(double t,const TargetTrajectories& target) {
  if(target.empty()) throw std::runtime_error("missing fixed world optical reference");
  std::size_t i=0;
  while(i+1<target.size() && target.timeTrajectory[i+1]<t) ++i;
  auto j=std::min(i+1,target.size()-1);
  double a=j==i ? 0. : std::clamp((t-target.timeTrajectory[i])/(target.timeTrajectory[j]-target.timeTrajectory[i]),0.,1.);
  const auto& p=target.stateTrajectory[i];const auto& q=target.stateTrajectory[j];
  if(p.size()!=7 || q.size()!=7) throw std::runtime_error("reference must be optical XYZ+XYZW");
  Eigen::Quaterniond r0(p(6),p(3),p(4),p(5)),r1(q(6),q(3),q(4),q(5));
  Pose out=Pose::Identity();out.topLeftCorner<3,3>()=r0.slerp(a,r1).toRotationMatrix();
  out.topRightCorner<3,1>()=(1-a)*p.head<3>()+a*q.head<3>();return out;
}
vector_t packed(const Pose& p) {
  vector_t x(7);Eigen::Quaterniond q(p.topLeftCorner<3,3>());
  x << p.topRightCorner<3,1>(),q.x(),q.y(),q.z(),q.w();return x;
}
Eigen::Matrix<double,6,1> error(const Pose& p,const Pose& goal) {
  Eigen::AngleAxisd angle(goal.topLeftCorner<3,3>().transpose()*p.topLeftCorner<3,3>());
  Eigen::Matrix<double,6,1> e;e << p.topRightCorner<3,1>()-goal.topRightCorner<3,1>(),angle.angle()*angle.axis();return e;
}

class CameraCost final:public StateCost {
 public:
  CameraCost(CameraModel model,double scale):model_(std::move(model)) { weight_ << 100.,100.,100.,20.,20.,20.;weight_*=scale; }
  CameraCost* clone() const override { return new CameraCost(*this); }
  scalar_t getValue(scalar_t t,const vector_t& x,const TargetTrajectories& target,const PreComputation&) const override {
    auto e=error(model_.camera(x),reference(t,target));return .5*e.dot(weight_.cwiseProduct(e)) + .5e-8*x.squaredNorm();
  }
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t t,const vector_t& x,const TargetTrajectories& target,const PreComputation& pre) const override {
    auto goal=reference(t,target);auto differential=model_.differential(x);auto e=error(differential.first,goal);
    matrix_t J=differential.second;Eigen::Vector3d angle=e.tail<3>();auto W=skew(angle);double theta=angle.norm();
    double c=theta<1e-5 ? 1./12. : (1.-.5*theta/std::tan(.5*theta))/(theta*theta);
    J.bottomRows(3)=(Eigen::Matrix3d::Identity()-.5*W+c*W*W)*goal.topLeftCorner<3,3>().transpose()*differential.second.bottomRows(3);
    ScalarFunctionQuadraticApproximation out;out.setZero(x.size());out.f=getValue(t,x,target,pre);
    out.dfdx=J.transpose()*weight_.cwiseProduct(e)+1e-8*x;
    // A consistent tiny state regularizer makes the rank-six GN Hessian
    // strictly positive despite floating-point roundoff in its nullspace.
    out.dfdxx=J.transpose()*weight_.asDiagonal()*J+1e-8*matrix_t::Identity(x.size(),x.size());return out;
  }
 private:CameraModel model_;Eigen::Matrix<double,6,1> weight_;
};

class InputCost final:public StateInputCost {
 public:
  explicit InputCost(int nu):R_(matrix_t::Identity(nu,nu)*.02) {}
  InputCost* clone() const override { return new InputCost(*this); }
  scalar_t getValue(scalar_t,const vector_t&,const vector_t& u,const TargetTrajectories&,const PreComputation&) const override { return .5*u.dot(R_*u); }
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t,const vector_t& x,const vector_t& u,const TargetTrajectories&,const PreComputation&) const override {
    ScalarFunctionQuadraticApproximation out;out.setZero(x.size(),u.size());out.f=.5*u.dot(R_*u);out.dfdu=R_*u;out.dfduu=R_;return out;
  }
 private:matrix_t R_;
};

