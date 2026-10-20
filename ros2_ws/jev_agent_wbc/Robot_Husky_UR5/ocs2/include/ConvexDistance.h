#pragma once
#include <Eigen/Dense>
#include <Eigen/QR>
#include <vector>
#include <stdexcept>
namespace jev_collision {
using matrix_t=Eigen::MatrixXd;using vector_t=Eigen::VectorXd;
struct Support {Eigen::Vector3d a,b,w;};
struct Distance {double lower=0.;Eigen::Vector3d a=Eigen::Vector3d::Zero(),b=Eigen::Vector3d::Zero();};
inline Distance distance(const std::vector<Eigen::Vector3d>& A,const std::vector<Eigen::Vector3d>& B){
  auto support=[&](const Eigen::Vector3d& d){auto a=A.front(),b=B.front();for(const auto& p:A)if(p.dot(d)>a.dot(d))a=p;for(const auto& p:B)if(p.dot(d)<b.dot(d))b=p;return Support{a,b,a-b};};
  auto first=support(Eigen::Vector3d::UnitX());std::vector<Support> simplex{first};Eigen::Vector3d closest=first.w;Distance result;result.a=first.a;result.b=first.b;
  for(int iteration=0;iteration<32;++iteration){double norm=closest.norm();if(norm<1e-10){result.lower=0.;return result;}auto next=support(-closest);result.lower=std::max(0.,closest.dot(next.w)/norm);if(norm-result.lower<1e-8)return result;
    auto candidates=simplex;candidates.push_back(next);double best=1e100;Eigen::Vector3d bestPoint;std::vector<Support> bestSimplex;Eigen::Vector3d bestA,bestB;
    for(unsigned mask=1;mask<(1u<<candidates.size());++mask){int count=__builtin_popcount(mask);if(count>4)continue;std::vector<int> ids;for(int i=0;i<candidates.size();++i)if(mask&(1u<<i))ids.push_back(i);
      matrix_t P(count,3);for(int i=0;i<count;++i)P.row(i)=candidates[ids[i]].w.transpose();matrix_t K=matrix_t::Zero(count+1,count+1);K.topLeftCorner(count,count)=P*P.transpose();K.topRightCorner(count,1).setOnes();K.bottomLeftCorner(1,count).setOnes();vector_t rhs=vector_t::Zero(count+1);rhs(count)=1.;vector_t weights=K.completeOrthogonalDecomposition().solve(rhs).head(count);
      if(!weights.allFinite() || weights.minCoeff()<-1e-9 || std::abs(weights.sum()-1.)>1e-6)continue;weights=weights.cwiseMax(0.);weights/=weights.sum();Eigen::Vector3d point=P.transpose()*weights;double squared=point.squaredNorm();if(squared>=best)continue;
      best=squared;bestPoint=point;bestSimplex.clear();bestA.setZero();bestB.setZero();for(int i=0;i<count;++i){bestA+=weights(i)*candidates[ids[i]].a;bestB+=weights(i)*candidates[ids[i]].b;if(weights(i)>1e-10)bestSimplex.push_back(candidates[ids[i]]);}
    }
    if(bestSimplex.empty())throw std::runtime_error("collision GJK simplex failed");simplex=std::move(bestSimplex);closest=bestPoint;result.a=bestA;result.b=bestB;
  }return result;
}
}
