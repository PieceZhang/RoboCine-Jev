#pragma once
#include "CameraMpcModel.h"
#include <map>

// Acquisition-time measured obstacles only. Robot shape ownership stays native.
class ObservedEnvironment {
  struct Body { int prefix; Pose home; };
  struct Shape { std::string body; Eigen::Vector3d center, extent; };
  struct Sphere { Eigen::Vector3d center, velocity; double radius, rate; };
  CameraModel camera_;
  std::map<std::string,Body> bodies_;
  std::vector<Shape> shapes_;
  std::vector<Sphere> spheres_;
  double stamp_;
 public:
  ObservedEnvironment(CameraModel camera,const Tree& model,const Tree& scene,double now)
    :camera_(std::move(camera)),stamp_(scene.get<double>("stamp_s")) {
    if(now<stamp_ || now>scene.get<double>("valid_until_s") || now-stamp_>.25)
      throw std::runtime_error("stale measured environment");
    for(const auto& entry:model.get_child("body_kinematics")) {
      const auto& b=entry.second;
      bodies_.emplace(b.get<std::string>("name"),Body{b.get<int>("prefix"),matrix(b.get_child("T_base_body"))});
    }
    for(const auto& entry:model.get_child("collision_shapes")) {
      const auto& s=entry.second;
      shapes_.push_back({s.get<std::string>("body"),array(s.get_child("center_body")),array(s.get_child("half_extents_body"))});
    }
    for(const auto& entry:scene.get_child("spheres")) {
      const auto& s=entry.second;
      double radius=s.get<double>("radius_m"),rate=s.get<double>("radius_rate_m_s");
      if(!std::isfinite(radius) || radius<=0 || !std::isfinite(rate) || rate<0)
        throw std::runtime_error("invalid observed obstacle radius");
      spheres_.push_back({array(s.get_child("center")),array(s.get_child("velocity")),radius,rate});
    }
  }
  void retainCostNeighborhood(double time,const vector_t& x) {
    auto values=evaluate(time,x,false).distance;std::vector<Sphere> kept;
    for(std::size_t j=0;j<spheres_.size();++j) {
      double minimum=1e9;
      for(std::size_t i=0;i<shapes_.size();++i)minimum=std::min(minimum,values(i*spheres_.size()+j));
      if(minimum<.45+2*spheres_[j].velocity.norm()+2*spheres_[j].rate)kept.push_back(spheres_[j]);
    }
    spheres_=std::move(kept);
  }
  struct Data { vector_t distance; matrix_t gradient; vector_t timeDerivative; };
  Data evaluate(double time,const vector_t& x,bool derivatives=true,double derivativeDistance=std::numeric_limits<double>::infinity())const {
    Pose base=Pose::Identity();
    base.topLeftCorner<3,3>()=Eigen::AngleAxisd(x(2),Eigen::Vector3d::UnitZ()).toRotationMatrix()*camera_.measuredTilt;
    base.topRightCorner<3,1>()<<x(0),x(1),camera_.height;
    std::vector<Pose> prefix(7,Pose::Identity());
    std::vector<Eigen::Vector3d> omega(6),velocity(6);
    for(int i=0;i<6;++i) {
      omega[i]=prefix[i].topLeftCorner<3,3>()*camera_.omega[i];
      velocity[i]=prefix[i].topLeftCorner<3,3>()*camera_.velocity[i]+prefix[i].topRightCorner<3,1>().cross(omega[i]);
      double a=x(3+i)-camera_.rest(i);auto w=skew(camera_.omega[i]);Pose e=Pose::Identity();
      e.topLeftCorner<3,3>()=Eigen::Matrix3d::Identity()+std::sin(a)*w+(1-std::cos(a))*w*w;
      e.topRightCorner<3,1>()=(a*Eigen::Matrix3d::Identity()+(1-std::cos(a))*w+(a-std::sin(a))*w*w)*camera_.velocity[i];
      prefix[i+1]=prefix[i]*e;
    }
    const int n=shapes_.size()*spheres_.size();
    Data out{vector_t::Zero(n),matrix_t::Zero(derivatives?n:0,x.size()),vector_t::Zero(n)};
    int k=0;
    for(const auto& s:shapes_) {
      const auto& b=bodies_.at(s.body);Pose T=base*prefix.at(b.prefix)*b.home;
      Eigen::Matrix3d R=T.topLeftCorner<3,3>();
      Eigen::Vector3d center=R*s.center+T.topRightCorner<3,1>();
      for(const auto& obstacle:spheres_) {
        Eigen::Vector3d c=obstacle.center+(time-stamp_)*obstacle.velocity;
        Eigen::Vector3d local=R.transpose()*(c-center);
        Eigen::Vector3d q=local.cwiseAbs()-s.extent;
        Eigen::Vector3d outside=q.cwiseMax(0.);
        double norm=outside.norm();
        Eigen::Vector3d localGradient=Eigen::Vector3d::Zero();
        double signedDistance;
        if(norm>1e-10) {
          for(int axis=0;axis<3;++axis)localGradient(axis)=(local(axis)>=0?1.:-1.)*outside(axis)/norm;
          signedDistance=norm;
        } else {
          Eigen::Index axis;q.maxCoeff(&axis);localGradient(axis)=local(axis)>=0?1.:-1.;
          signedDistance=q(axis);
        }
        Eigen::Vector3d normal=-R*localGradient;
        Eigen::Vector3d closest=center+R*local.cwiseMax(-s.extent).cwiseMin(s.extent);
        out.distance(k)=signedDistance-obstacle.radius-std::max(0.,time-stamp_)*obstacle.rate;
        if(!derivatives || out.distance(k)>=derivativeDistance){++k;continue;}
        out.gradient(k,0)=normal.x();out.gradient(k,1)=normal.y();
        out.gradient(k,2)=normal.dot(Eigen::Vector3d::UnitZ().cross(closest-base.topRightCorner<3,1>()));
        Eigen::Vector3d pbase=base.topLeftCorner<3,3>().transpose()*(closest-base.topRightCorner<3,1>());
        for(int j=0;j<b.prefix;++j)
          out.gradient(k,3+j)=normal.dot(base.topLeftCorner<3,3>()*(omega[j].cross(pbase)+velocity[j]));
        out.timeDerivative(k)=-normal.dot(obstacle.velocity)-obstacle.rate;
        ++k;
      }
    }
    return out;
  }
};

class LocalEnvironmentCost final:public StateCost {
  ObservedEnvironment model_;
  // Anticipatory preference only; the worker retains its 25 mm hard gate.
  static constexpr double margin_=.14;
 public:
  LocalEnvironmentCost(const ObservedEnvironment& model,double stamp,const vector_t& x):model_(model) {
    model_.retainCostNeighborhood(stamp,x);
  }
  LocalEnvironmentCost* clone()const override{return new LocalEnvironmentCost(*this);}
  scalar_t getValue(scalar_t time,const vector_t& x,const TargetTrajectories&,const PreComputation&)const override {
    auto data=model_.evaluate(time,x,false);
    return 20000*(vector_t::Constant(data.distance.size(),margin_)-data.distance).cwiseMax(0.).squaredNorm();
  }
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t time,const vector_t& x,const TargetTrajectories&,const PreComputation&)const override {
    auto data=model_.evaluate(time,x,true,margin_);
    ScalarFunctionQuadraticApproximation out;out.setZero(x.size());
    for(int i=0;i<data.distance.size();++i) {
      double r=margin_-data.distance(i);if(r<=0.)continue;
      vector_t gradient=-data.gradient.row(i).transpose();
      out.f+=20000*r*r;out.dfdx.noalias()+=40000*r*gradient;
      out.dfdxx.noalias()+=40000*gradient*gradient.transpose();
    }
    return out;
  }
};
