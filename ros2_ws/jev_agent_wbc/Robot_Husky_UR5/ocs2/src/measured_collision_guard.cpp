#include "ConvexDistance.h"
#include <boost/property_tree/json_parser.hpp>
#include <map>
#include <string>
#include <memory>
using Tree=boost::property_tree::ptree;
using RowPose=Eigen::Matrix<double,4,4,Eigen::RowMajor>;
struct Shape{int body;Eigen::Vector3d center,extent;std::vector<Eigen::Vector3d> vertices;};
struct Guard{int bodies;std::vector<Shape> shapes;std::vector<std::pair<int,int>> pairs;};
thread_local std::string last_error;
Eigen::Vector3d vector3(const Tree& t){if(t.size()!=3)throw std::runtime_error("expected vector3");Eigen::Vector3d v;int i=0;for(const auto& x:t)v(i++)=x.second.get_value<double>();if(!v.allFinite())throw std::runtime_error("nonfinite geometry");return v;}
extern "C" void* jev_guard_create(const char* path)try{
 Tree model;boost::property_tree::read_json(path,model);auto g=std::make_unique<Guard>();std::map<std::string,int> bodies;
 for(const auto& x:model.get_child("body_kinematics")){int index=bodies.size();bodies.emplace(x.second.get<std::string>("name"),index);}g->bodies=bodies.size();
 for(const auto& x:model.get_child("collision_shapes")){const auto& item=x.second;Shape s{bodies.at(item.get<std::string>("body")),vector3(item.get_child("center_body")),vector3(item.get_child("half_extents_body")),{}};for(const auto& v:item.get_child("vertices_body"))s.vertices.push_back(vector3(v.second));g->shapes.push_back(std::move(s));}
 for(const auto& x:model.get_child("collision_pairs")){auto p=x.second.begin();int a=p++->second.get_value<int>(),b=p->second.get_value<int>();if(a<0 || b<0 || a>=g->shapes.size() || b>=g->shapes.size())throw std::runtime_error("bad collision pair");g->pairs.emplace_back(a,b);}return g.release();
}catch(const std::exception& e){last_error=e.what();return nullptr;}
extern "C" int jev_guard_check(void* handle,const double* matrices,int count,double* minimum,int* pair_a,int* pair_b)try{
 auto& g=*static_cast<Guard*>(handle);if(count!=g.bodies)throw std::runtime_error("missing measured physical bodies");
 struct Box{Eigen::Vector3d center,extent;Eigen::Matrix3d axes;};std::vector<Box> boxes;std::vector<std::vector<Eigen::Vector3d>> vertices;
 for(const auto& s:g.shapes){Eigen::Map<const RowPose> T(matrices+s.body*16);if(!T.allFinite() || std::abs(T(3,3)-1)>1e-8)throw std::runtime_error("nonfinite measured body pose");Eigen::Matrix3d R=T.topLeftCorner<3,3>();Eigen::Vector3d t=T.topRightCorner<3,1>();boxes.push_back({R*s.center+t,s.extent,R});std::vector<Eigen::Vector3d> points;for(const auto& v:s.vertices)points.push_back(R*v+t);vertices.push_back(std::move(points));}
 *minimum=1e9;*pair_a=-1;*pair_b=-1;
 for(const auto& pair:g.pairs){int ia=pair.first,ib=pair.second;const auto& a=boxes[ia];const auto& b=boxes[ib];auto delta=b.center-a.center;double distance=-1e9;
  auto axis=[&](Eigen::Vector3d n){double norm=n.norm();if(norm<1e-9)return;n/=norm;distance=std::max(distance,std::abs(n.dot(delta))-(a.axes.transpose()*n).cwiseAbs().dot(a.extent)-(b.axes.transpose()*n).cwiseAbs().dot(b.extent));};
  for(int i=0;i<3;++i){axis(a.axes.col(i));axis(b.axes.col(i));for(int j=0;j<3;++j)axis(a.axes.col(i).cross(b.axes.col(j)));}
  if(distance<.025)distance=std::max(distance,jev_collision::distance(vertices[ia],vertices[ib]).lower);
  if(distance<*minimum){*minimum=distance;*pair_a=ia;*pair_b=ib;}
 }return 0;
}catch(const std::exception& e){last_error=e.what();return -1;}
extern "C" const char* jev_guard_error(){return last_error.c_str();}
extern "C" void jev_guard_destroy(void* handle){delete static_cast<Guard*>(handle);}
