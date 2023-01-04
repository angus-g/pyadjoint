import pytest
pytest.importorskip("firedrake")

from numpy.testing import assert_allclose
from firedrake import *
from firedrake_adjoint import *
from pyadjoint import LinMoreOptimiser

# This is the classic way
rol_params = {
    'General': {
        'Print Verbosity': 1,
        'Output Level': 1,
        'Secant': {'Type': 'Limited-Memory BFGS',
                   'Maximum Storage': 5,
                   'Use as Hessian': True,
                   "Barzilai-Borwein": 1
                   },
    },
    'Step': {
        'Type': 'Trust Region',
        'Trust Region': {
            "Subproblem Model": "Lin-More",
        },
    },
    'Status Test': {
        'Constraint Tolerance': 0,
        'Gradient Tolerance': 1e-5,
        'Iteration Limit': 50,
    }
}

results = []


def _simple_helmholz_model(V, source):
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(v), grad(u)) * dx + 100.0 * v * u * dx - v * source * dx
    solve(F == 0, u)
    return u


def test_simple_inversion_linmore():
    """Test inversion of source term in helmholze eqn."""
    mesh = UnitIntervalMesh(10)

    V = FunctionSpace(mesh, "CG", 1)
    source_ref = Function(V)
    x = SpatialCoordinate(mesh)
    source_ref.interpolate(cos(pi * x**2))

    # compute reference solution
    with stop_annotating():
        u_ref = _simple_helmholz_model(V, source_ref)

    # now rerun annotated model with zero source
    source = Function(V)
    c = Control(source)
    u = _simple_helmholz_model(V, source)

    J = assemble(1e6 * (u - u_ref)**2 * dx)
    rf = ReducedFunctional(J, c)

    class rf_cb(object):
        def __init__(self):
            self.result = []

        def __call__(self):
            self.result.append(J.block_variable.checkpoint)

    values_holder = rf_cb()

    # Set up bounds, which will later be used to
    # enforce boundary conditions in inversion:
    T_lb = Function(V, name="LB_Temperature")
    T_ub = Function(V, name="UB_Temperature")
    T_lb.assign(-1.5)
    T_ub.assign(1.5)

    # Optimise using ROL
    minp = MinimizationProblem(rf, bounds=(T_lb, T_ub))
    lin_more = LinMoreOptimiser(minp, rol_params, callback=values_holder, checkpoint=False)
    lin_more.run()
    assert_allclose(
        lin_more.rol_solver.rolvector.dat[0].dat.data,
        source_ref.dat.data,
        rtol=1e-2)
