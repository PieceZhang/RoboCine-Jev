#pragma once
#include "CameraMpcModel.h"
#include <map>
#include <Eigen/QR>

#include "ConvexDistance.h"

class CollisionCost final : public StateCost {
  struct Body{int prefix;Pose home;};
  struct Shape{std::string body;Eigen::Vector3d center,extent;std::vector<Eigen::Vector3d> vertices;};
  CameraModel camera_;std::map<std::string,Body> bodies_;std::vector<Shape> shapes_;std::vector<std::pair<int,int>> pairs_;
  double weight_;
 public:
  CollisionCost(CameraModel camera,const Tree& model,double weight=5000.):camera_(std::move(camera)),weight_(weight){
    for(const auto& entry:model.get_child("body_kinematics")){const auto& x=entry.second;bodies_.emplace(x.get<std::string>("name"),Body{x.get<int>("prefix"),matrix(x.get_child("T_base_body"))});}
    for(const auto& entry:model.get_child("collision_shapes")){const auto& x=entry.second;Shape s{x.get<std::string>("body"),array(x.get_child("center_body")),array(x.get_child("half_extents_body")),{}};for(const auto& v:x.get_child("vertices_body"))s.vertices.push_back(array(v.second));shapes_.push_back(std::move(s));}
    for(const auto& entry:model.get_child("collision_pairs")){auto p=array(entry.second);if(p.size()!=2)throw std::runtime_error("invalid collision pair");pairs_.emplace_back(int(p(0)),int(p(1)));}
  }
  CollisionCost* clone()const override{return new CollisionCost(*this);}
  void retainCostNeighborhood(const vector_t& x) {
    auto data=residual(x,true,.40);
    std::vector<std::pair<int,int>> kept;
    for(std::size_t i=0;i<pairs_.size();++i) {
      const auto& pair=pairs_[i];
      const auto& a=bodies_.at(shapes_[pair.first].body);
      const auto& b=bodies_.at(shapes_[pair.second].body);
      if(a.prefix!=b.prefix && data.first(i)>0.)kept.push_back(pair);
    }
    pairs_=std::move(kept);
  }
  std::pair<vector_t,matrix_t> residual(const vector_t& x,bool derivative,double margin=.025)const{
    std::vector<Pose> prefix(7,Pose::Identity());std::vector<Eigen::Vector3d> omega(6),velocity(6);
    for(int i=0;i<6;++i){omega[i]=prefix[i].topLeftCorner<3,3>()*camera_.omega[i];velocity[i]=prefix[i].topLeftCorner<3,3>()*camera_.velocity[i]+prefix[i].topRightCorner<3,1>().cross(omega[i]);double a=x(3+i)-camera_.rest(i);auto w=skew(camera_.omega[i]);Pose e=Pose::Identity();e.topLeftCorner<3,3>()=Eigen::Matrix3d::Identity()+std::sin(a)*w+(1-std::cos(a))*w*w;e.topRightCorner<3,1>()=(a*Eigen::Matrix3d::Identity()+(1-std::cos(a))*w+(a-std::sin(a))*w*w)*camera_.velocity[i];prefix[i+1]=prefix[i]*e;}
    struct Box{Eigen::Vector3d center,extent;Eigen::Matrix3d axes;};std::vector<Box> boxes;std::vector<std::vector<Eigen::Vector3d>> vertices;
    for(const auto& s:shapes_){const auto& b=bodies_.at(s.body);Pose T=prefix.at(b.prefix)*b.home;boxes.push_back({T.topLeftCorner<3,3>()*s.center+T.topRightCorner<3,1>(),s.extent,T.topLeftCorner<3,3>()});std::vector<Eigen::Vector3d> points;for(const auto& v:s.vertices)points.push_back(T.topLeftCorner<3,3>()*v+T.topRightCorner<3,1>());vertices.push_back(std::move(points));}
    vector_t r=vector_t::Zero(pairs_.size());matrix_t J=matrix_t::Zero(pairs_.size(),x.size());
    for(std::size_t k=0;k<pairs_.size();++k){int ia=pairs_[k].first,ib=pairs_[k].second;const auto& a=boxes.at(ia);const auto& b=boxes.at(ib);double bound=-1e9;auto delta=b.center-a.center;
      auto axis=[&](Eigen::Vector3d n){double norm=n.norm();if(norm<1e-9)return;n/=norm;bound=std::max(bound,std::abs(n.dot(delta))-(a.axes.transpose()*n).cwiseAbs().dot(a.extent)-(b.axes.transpose()*n).cwiseAbs().dot(b.extent));};
      for(int i=0;i<3;++i){axis(a.axes.col(i));axis(b.axes.col(i));for(int j=0;j<3;++j)axis(a.axes.col(i).cross(b.axes.col(j)));}if(bound>=margin)continue;
      auto closest=jev_collision::distance(vertices[ia],vertices[ib]);bound=std::max(bound,closest.lower);r(k)=std::max(0.,margin-bound);if(!derivative || r(k)==0.)continue;
      Eigen::Vector3d difference=closest.a-closest.b;double norm=difference.norm();if(norm<1e-10)continue;Eigen::Vector3d normal=difference/norm;
      const auto& ba=bodies_.at(shapes_[ia].body);const auto& bb=bodies_.at(shapes_[ib].body);
      for(int j=0;j<6;++j){Eigen::Vector3d da=Eigen::Vector3d::Zero(),db=Eigen::Vector3d::Zero();if(j<ba.prefix)da=omega[j].cross(closest.a)+velocity[j];if(j<bb.prefix)db=omega[j].cross(closest.b)+velocity[j];J(k,3+j)=-normal.dot(da-db);}
    }return {r,J};
  }
  scalar_t getValue(scalar_t,const vector_t& x,const TargetTrajectories&,const PreComputation&)const override{return weight_*residual(x,false).first.squaredNorm();}
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t,const vector_t& x,const TargetTrajectories&,const PreComputation&)const override{auto r=residual(x,true);ScalarFunctionQuadraticApproximation out;out.setZero(x.size());out.f=weight_*r.first.squaredNorm();out.dfdx=2*weight_*r.second.transpose()*r.first;out.dfdxx=2*weight_*r.second.transpose()*r.second;return out;}
};

// Re-linearize exact convex-hull distances at every fresh physical state.
// The final native rollout is checked again against exact hulls before release.
class LocalCollisionCost final : public StateCost {
  vector_t anchor_,r_;matrix_t J_;
 public:
  LocalCollisionCost(const CollisionCost& exact,const vector_t& state):anchor_(state){
    auto data=exact.residual(state,true);r_=data.first;J_=data.second;
  }
  LocalCollisionCost* clone()const override{return new LocalCollisionCost(*this);}
  scalar_t getValue(scalar_t,const vector_t& x,const TargetTrajectories&,const PreComputation&)const override{return 5000*(r_+J_*(x-anchor_)).cwiseMax(0.).squaredNorm();}
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t,const vector_t& x,const TargetTrajectories&,const PreComputation&)const override{
    vector_t r=r_+J_*(x-anchor_);matrix_t J=J_;for(int i=0;i<r.size();++i)if(r(i)<=0.){r(i)=0.;J.row(i).setZero();}
    ScalarFunctionQuadraticApproximation out;out.setZero(x.size());out.f=5000*r.squaredNorm();out.dfdx=10000*J.transpose()*r;out.dfdxx=10000*J.transpose()*J;return out;
  }
};

// Numerical linearizations along the last solve provide early repulsion when
// a pair is safe at the acquisition pose but approaches within the horizon.
// These are optimization seeds only; the fresh final policy still passes the
// independent full convex-hull check before any command is released.
class PredictiveLocalCollisionCost final : public StateCost {
  vector_array_t anchors_,residuals_;
  std::vector<matrix_t> jacobians_;
  double weight_;
 public:
  PredictiveLocalCollisionCost(const CollisionCost& exact,const vector_array_t& anchors,double weight)
      :anchors_(anchors),weight_(weight) {
    for(const auto& anchor:anchors_) {
      auto data=exact.residual(anchor,true,.12);
      residuals_.push_back(data.first-vector_t::Constant(data.first.size(),.095));
      jacobians_.push_back(std::move(data.second));
    }
  }
  PredictiveLocalCollisionCost* clone()const override{return new PredictiveLocalCollisionCost(*this);}
  std::pair<vector_t,matrix_t> residual(const vector_t& x)const {
    vector_t r=vector_t::Zero(residuals_.front().size());
    matrix_t J=matrix_t::Zero(r.size(),x.size());
    for(std::size_t k=0;k<anchors_.size();++k) {
      vector_t candidate=residuals_[k]+jacobians_[k]*(x-anchors_[k]);
      for(int i=0;i<r.size();++i)if(candidate(i)>r(i)) {
        r(i)=candidate(i);J.row(i)=jacobians_[k].row(i);
      }
    }
    return {r,J};
  }
  scalar_t getValue(scalar_t,const vector_t& x,const TargetTrajectories&,const PreComputation&)const override {
    return weight_*residual(x).first.squaredNorm();
  }
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t,const vector_t& x,const TargetTrajectories&,const PreComputation&)const override {
    auto data=residual(x);ScalarFunctionQuadraticApproximation out;out.setZero(x.size());
    out.f=weight_*data.first.squaredNorm();
    out.dfdx=2*weight_*data.second.transpose()*data.first;
    out.dfdxx=2*weight_*data.second.transpose()*data.second;
    return out;
  }
};
