#pragma once

#include <cmath>
#include <stdexcept>
#include <ocs2_core/dynamics/SystemDynamicsBase.h>

namespace jev {

/** Actual OCS2 dynamics extension; this class contains no optimizer.
 * State: world [x,y,yaw,q_arm]. Inputs: body [v,w,dq_arm] or
 * [vx,vy,w,dq_arm]. The omni variant is deliberately not upstream model type 1.
 */
class WheeledDynamics final : public ocs2::SystemDynamicsBase {
 public:
  explicit WheeledDynamics(bool omni, int armDim = 6, double lateralYawGain = 0.)
      : omni_(omni), armDim_(armDim), lateralYawGain_(lateralYawGain) {
    if (armDim < 1) throw std::invalid_argument("armDim must be positive");
    if(!std::isfinite(lateralYawGain) || std::abs(lateralYawGain)>.35)
      throw std::invalid_argument("invalid measured lateral yaw gain");
  }
  WheeledDynamics* clone() const override { return new WheeledDynamics(*this); }
  using SystemDynamicsBase::computeFlowMap;
  using SystemDynamicsBase::linearApproximation;
  int stateDim() const { return 3 + armDim_; }
  int inputDim() const { return (omni_ ? 3 : 2) + armDim_; }

  ocs2::vector_t computeFlowMap(ocs2::scalar_t, const ocs2::vector_t& x,
                              const ocs2::vector_t& u, const ocs2::PreComputation&) override {
    check(x, u);
    const double c = std::cos(x(2)), s = std::sin(x(2));
    const double vx = u(0), vy = omni_ ? u(1) : lateralYawGain_*u(1);
    ocs2::vector_t f(stateDim());
    f << c*vx-s*vy, s*vx+c*vy, u(omni_ ? 2 : 1), u.tail(armDim_);
    return f;
  }

  ocs2::VectorFunctionLinearApproximation linearApproximation(
      ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u,
      const ocs2::PreComputation& preComp) override {
    check(x, u);
    ocs2::VectorFunctionLinearApproximation a;
    a.f = computeFlowMap(t, x, u, preComp);
    a.dfdx = ocs2::matrix_t::Zero(stateDim(), stateDim());
    a.dfdu = ocs2::matrix_t::Zero(stateDim(), inputDim());
    const double c = std::cos(x(2)), s = std::sin(x(2));
    const double vx = u(0), vy = omni_ ? u(1) : lateralYawGain_*u(1);
    a.dfdx(0,2) = -s*vx-c*vy;
    a.dfdx(1,2) = c*vx-s*vy;
    a.dfdu(0,0) = c;
    a.dfdu(1,0) = s;
    if (omni_) {
      a.dfdu(0,1) = -s;
      a.dfdu(1,1) = c;
    } else {
      a.dfdu(0,1)=-s*lateralYawGain_;
      a.dfdu(1,1)=c*lateralYawGain_;
    }
    a.dfdu(2,omni_ ? 2 : 1) = 1.;
    a.dfdu.bottomRightCorner(armDim_,armDim_).setIdentity();
    return a;
  }

 private:
  void check(const ocs2::vector_t& x, const ocs2::vector_t& u) const {
    if (x.size() != stateDim() || u.size() != inputDim() || !x.allFinite() || !u.allFinite())
      throw std::invalid_argument("invalid state/input for the selected base actuation");
  }
  bool omni_;
  int armDim_;
  double lateralYawGain_;
};
}  // namespace jev
