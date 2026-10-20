import numpy as np
import pytest
from scipy.optimize import minimize,LinearConstraint
from jev_agent_wbc.small_convex_qp import solve


@pytest.mark.parametrize('seed',range(12))
def test_dense_faces_match_independent_constrained_solver(seed):
    rng=np.random.default_rng(seed);M=rng.normal(size=(2,2));H=M.T@M+.1*np.eye(2)
    g=rng.normal(size=2)*5;A=rng.normal(size=(220,2));A/=np.linalg.norm(A,axis=1)[:,None]
    center=rng.normal(size=2);radii=rng.uniform(.01,.6,220)
    lo=np.full(220,-np.inf);hi=A@center+radii
    answer=solve(H,g,A,lo,hi)
    checked=minimize(lambda x:.5*x@H@x+g@x,center,jac=lambda x:H@x+g,
        constraints=LinearConstraint(A,lo,hi),method='SLSQP',options=dict(ftol=1e-12,maxiter=200))
    assert checked.success
    assert np.max(A@answer-hi)<1e-8
    np.testing.assert_allclose(answer,checked.x,atol=2e-7)


def test_nearly_parallel_faces_and_empty_intersection():
    A=np.array([[1.,0.],[1.,1e-11],[0.,1.]])
    answer=solve(np.eye(2),np.array([-3.,4.]),A,np.array([0.,0.,-.2]),np.array([.1,.1,.2]))
    np.testing.assert_allclose(answer,[.1,-.2],atol=1e-9)
    with pytest.raises(RuntimeError):
        solve(np.eye(2),np.zeros(2),np.array([[1.,0.],[1.,0.]]),np.array([1.,-np.inf]),np.array([np.inf,0.]))
