#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include "WheeledDynamics.h"

int main() {
  constexpr bool omni = JEV_OMNI_BASE;
  jev::WheeledDynamics model(omni);
  std::mt19937 random(20261020);
  std::uniform_real_distribution<double> distribution(-2.,2.);
  double maxError = 0., maxLateralResidual = 0.;
  constexpr double h = 1e-6;
  for (int sample = 0; sample < 128; ++sample) {
    ocs2::vector_t x(9), u(model.inputDim());
    for (int i = 0; i < x.size(); ++i) x(i) = distribution(random);
    for (int i = 0; i < u.size(); ++i) u(i) = distribution(random);
    auto a = model.linearApproximation(0.,x,u);
    for (int i = 0; i < x.size(); ++i) {
      auto plus=x, minus=x; plus(i)+=h; minus(i)-=h;
      ocs2::vector_t numeric=(model.computeFlowMap(0.,plus,u)-model.computeFlowMap(0.,minus,u))/(2*h);
      maxError=std::max(maxError,(numeric-a.dfdx.col(i)).cwiseAbs().maxCoeff());
    }
    for (int i = 0; i < u.size(); ++i) {
      auto plus=u, minus=u; plus(i)+=h; minus(i)-=h;
      ocs2::vector_t numeric=(model.computeFlowMap(0.,x,plus)-model.computeFlowMap(0.,x,minus))/(2*h);
      maxError=std::max(maxError,(numeric-a.dfdu.col(i)).cwiseAbs().maxCoeff());
    }
    const double lateral=-std::sin(x(2))*a.f(0)+std::cos(x(2))*a.f(1);
    maxLateralResidual=std::max(maxLateralResidual,std::abs(lateral-(omni ? u(1) : 0.)));
    std::unique_ptr<ocs2::SystemDynamicsBase> copy(model.clone());
    if (!copy->computeFlowMap(0.,x,u).isApprox(a.f,1e-14)) return 2;
  }
  bool rejectsWrongInput=false;
  try { model.computeFlowMap(0.,ocs2::vector_t::Zero(9),ocs2::vector_t::Zero(omni ? 8 : 9)); }
  catch (const std::invalid_argument&) { rejectsWrongInput=true; }
  const bool passed=maxError<1e-8 && maxLateralResidual<1e-12 && rejectsWrongInput;
  std::cout << "{\"framework\":\"actual_ocs2_core\",\"scope\":\"compiled dynamics and derivatives only\","
            << "\"state_dim\":9,\"input_dim\":" << model.inputDim()
            << ",\"omni\":" << (omni ? "true" : "false")
            << ",\"samples\":128,\"max_derivative_error\":" << maxError
            << ",\"max_body_lateral_residual\":" << maxLateralResidual
            << ",\"mpc_executed\":false,\"passed\":" << (passed ? "true" : "false") << "}\n";
  return passed ? 0 : 2;
}
