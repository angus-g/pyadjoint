from numpy.testing import assert_allclose
from firedrake import *
from firedrake_adjoint import *
from pyadjoint import LinMoreOptimiser, ROLCheckpointManager

# This is the classic way
rol_params = {
    'General': {
        'Print Verbosity': 0,
        'Output Level': 0,
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
        'Gradient Tolerance': 1e-3,
        'Step Tolerance': 1e-3,
        'Iteration Limit': 10,
    }
}

results = []


def _make_checkpointable_mesh():
    # mesh
    mesh = UnitIntervalMesh(10)

    # writing out mesh
    with CheckpointFile("mesh.h5", mode='w') as chptfi:
        chptfi.save_mesh(mesh)


def _simple_helmholz_model(V, source):
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(v), grad(u)) * dx + 100.0 * v * u * dx - v * source * dx
    solve(F == 0, u)
    return u


def _solve_minimisation(num_iteration, restart=None):
    """Test inversion of source term in helmholze eqn."""

    # clean up tape
    tape = get_working_tape()
    tape.clear_tape()

    with CheckpointFile('mesh.h5', mode='r') as fi:
        mesh = fi.load_mesh("firedrake_default")

    ROLCheckpointManager.set_checkpoint_dir('./checkpoint_dir', cleanup=False)
    ROLCheckpointManager.set_mesh('mesh.h5', "firedrake_default")

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

    # Set up bounds, which will later be used to
    # enforce boundary conditions in inversion:
    T_lb = Function(V, name="LB_Temperature")
    T_ub = Function(V, name="UB_Temperature")
    T_lb.assign(-1.0)
    T_ub.assign(1.0)

    # Optimise using ROL
    minp = MinimizationProblem(rf, bounds=(T_lb, T_ub))

    # replacing the num of iterations to continue
    rol_params.get('Status Test')['Iteration Limit'] = num_iteration

    values_holder = rf_cb()

    lin_more = LinMoreOptimiser(minp, rol_params, callback=values_holder, checkpoint=True)

    if restart:
        lin_more.reload(iteration=restart)

    lin_more.run()
    return values_holder


def test_linmore_checkpointing():
    """
        Testing checking pointing with ROL2.0
    """
    _make_checkpointable_mesh()
    maximum_iteartion = 10
    complete_result = _solve_minimisation(maximum_iteartion)
    for num_it in range(2, maximum_iteartion, 2):
        restart_result = _solve_minimisation(
            maximum_iteartion - num_it, restart=num_it)
        assert_allclose(
            complete_result.result[num_it:],
            restart_result.result,
            rtol=1e-5)
