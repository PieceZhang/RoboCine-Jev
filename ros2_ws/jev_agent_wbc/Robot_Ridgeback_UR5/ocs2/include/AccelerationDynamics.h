#pragma once
#include "WheeledDynamics.h"

namespace jev {
// Native OCS2 input-rate augmentation. The tail is the last APPLIED velocity
// command and the input its derivative; this is not a contact/PD dynamic model.
// Pose/joint state and real velocities are measured separately by the executor.
class AccelerationDynamics final : public ocs2::SystemDynamicsBase {
 public:
  explicit AccelerationDynamics(bool omni) : flow_(omni), nu_(omni ? 9 : 8) {}
  AccelerationDynamics* clone() const override { return new AccelerationDynamics(*this); }
  using SystemDynamicsBase::computeFlowMap;
  using SystemDynamicsBase::linearApproximation;
  ocs2::vector_t computeFlowMap(ocs2::scalar_t t, const ocs2::vector_t& x,
      const ocs2::vector_t& a, const ocs2::PreComputation& pre) override {
    if(x.size()!=9+nu_ || a.size()!=nu_ || !x.allFinite() || !a.allFinite())
      throw std::invalid_argument("invalid acceleration-model state/input");
    ocs2::vector_t result(9+nu_);
    result << flow_.computeFlowMap(t,x.head(9),x.tail(nu_),pre),a;
    return result;
  }
  ocs2::VectorFunctionLinearApproximation linearApproximation(ocs2::scalar_t t,
      const ocs2::vector_t& x,const ocs2::vector_t& a,const ocs2::PreComputation& pre) override {
    auto result=ocs2::VectorFunctionLinearApproximation();
    result.f=computeFlowMap(t,x,a,pre);
    result.dfdx=ocs2::matrix_t::Zero(x.size(),x.size());
    result.dfdu=ocs2::matrix_t::Zero(x.size(),nu_);
    auto kinematic=flow_.linearApproximation(t,x.head(9),x.tail(nu_),pre);
    result.dfdx.topLeftCorner(9,9)=kinematic.dfdx;
    result.dfdx.topRightCorner(9,nu_)=kinematic.dfdu;
    result.dfdu.bottomRows(nu_).setIdentity();
    return result;
  }
 private:WheeledDynamics flow_;int nu_;
};
}
