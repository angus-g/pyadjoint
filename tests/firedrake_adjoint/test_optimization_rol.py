import pytest
from pathlib import Path

from firedrake import *
from firedrake_adjoint import *


@pytest.fixture
def base_path():
    return Path(__file__).parent


def setup_problem(path, n=20):
    mesh = Mesh(str(path / f"square_{n}.msh"))
    V = FunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "DG", 0)
    x = SpatialCoordinate(mesh)

    f = Function(W)
    f.interpolate(x[0] + x[1])
    u = Function(V, name="State")
    v = TestFunction(V)

    F = (inner(grad(u), grad(v)) - f*v) * dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, u, bc)

    w = sin(pi*x[0]) * sin(pi*x[1])
    d = w / (2*pi**2)
    alpha = Constant(1e-3)
    J = assemble((0.5 * inner(u-d, u-d)) * dx(degree=3) + alpha / 2*f**2 * dx(degree=3))

    control = Control(f)
    rf = ReducedFunctional(J, control)
    params = {
        "Step": {
            "Type": "Line Search",
        },
        "Status Test": {
            "Gradient Tolerance": 1e-11,
            "Iteration Limit": 20,
        },
    }

    return rf, params, w, alpha


def test_finds_analytical_solution(base_path):
    rf, params, w, alpha = setup_problem(base_path)
    problem = MinimizationProblem(rf)
    solver = ROLSolver(problem, params, inner_product="L2")
    sol = solver.solve()

    f_analytic = 1 / (1 + alpha*4*pow(pi, 4)) * w

    assert errornorm(f_analytic, sol) < 0.02


def test_bounds_work_sensibly(base_path):
    rf, params, w, alpha = setup_problem(base_path)
    lower = 0
    upper = 0.5

    problem = MinimizationProblem(rf, bounds=(lower, upper))
    solver = ROLSolver(problem, params, inner_product="L2")
    sol1 = solver.solve().copy(deepcopy=True)
    f = rf.controls[0]
    V = f.function_space()

    lower = interpolate(Constant(lower), V)
    upper = interpolate(Constant(upper), V)
    problem = MinimizationProblem(rf, bounds=(lower, upper))
    solver = ROLSolver(problem, params, inner_product="L2")
    solver.rolvector.scale(0.0)
    sol2 = solver.solve()

    assert errornorm(sol1, sol2) < 1e-7


@pytest.mark.parametrize("contype", ["eq", "ineq"])
def test_ufl_constraint_works_sensibly(contype, base_path):
    rf, params, w, alpha = setup_problem(base_path, n=7)

    params = {
        'General': {
            'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 10}},
        'Step': {
            'Type': 'Augmented Lagrangian',
            'Line Search': {
                'Descent Method': {
                    'Type': 'Quasi-Newton Step'
                }
            },
            'Augmented Lagrangian': {
                'Subproblem Step Type': 'Line Search',
                'Subproblem Iteration Limit': 20
            }
        },
        'Status Test': {
            'Gradient Tolerance': 1e-7,
            'Iteration Limit': 15
        }
    }

    f = rf.controls[0]
    V = f.function_space()
    vol = 0.3
    econ = UFLEqualityConstraint((vol - f.control**2)*dx, f)
    icon = UFLInequalityConstraint((vol - f.control**2)*dx, f)
    bounds = (interpolate(Constant(0.0), V), interpolate(Constant(0.7), V))


    if contype == "eq":
        print("Run with equality constraint")
        problem = MinimizationProblem(rf, constraints=[econ])
    elif contype == "ineq":
        print("Run with inequality constraint")
        problem = MinimizationProblem(rf, constraints=[icon])
        return
    else:
        raise NotImplementedError

    solver = ROLSolver(problem, params, inner_product="L2")

    econ, emul = solver.constraints[0]
    icon, imul = solver.constraints[1]

    x = solver.rolvector
    v = x.clone()
    v.dat[0].interpolate(Constant(1.0))
    u = v.clone()
    u.plus(v)


    if len(econ)>0:
        jv = emul[0].clone()
        jv.dat[0].assign(1.0)
        res0 = econ[0].checkApplyJacobian(x, v, jv, 5, 1)
        res1 = econ[0].checkAdjointConsistencyJacobian(jv, v, x)
        res2 = econ[0].checkApplyAdjointHessian(x, jv, u, v, 5, 1)

        for i in range(1, len(res0)):
            assert res0[i][3] < 0.15 * res0[i-1][3]
        assert res1 < 1e-10
        assert all(r[3] < 1e-10 for r in res2)

    if len(icon)>0:
        jv = imul[0].clone()
        jv.dat[0].assign(1.0)
        res0 = icon[0].checkApplyJacobian(x, v, jv, 5, 1)
        res1 = icon[0].checkAdjointConsistencyJacobian(jv, v, x)
        res2 = icon[0].checkApplyAdjointHessian(x, jv, u, v, 5, 1)

        for i in range(1, len(res0)):
            assert res0[i][3] < 0.15 * res0[i-1][3]
        assert res1 < 1e-10
        assert all(r[3] < 1e-10 for r in res2)

    sol1 = solver.solve().copy(deepcopy=True)
    if contype == "eq":
        assert(abs(assemble(sol1**2 * dx) - vol) < 1e-5)
    elif contype == "ineq":
        assert(assemble(sol1**2 * dx) < vol + 1e-5)
    else:
        raise NotImplementedError
