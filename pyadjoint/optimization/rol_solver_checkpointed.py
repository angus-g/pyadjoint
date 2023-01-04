"""
    Checkpointed optimisation with ROL2.0
"""
import os
import shutil

from ..tape import no_annotations
from mpi4py import MPI
from .rol_solver import (
    ROLVector, ROLObjective, ROLSolver)
from .optimization_solver import OptimizationSolver
from firedrake.checkpointing import CheckpointFile
from firedrake import utils


try:
    import ROL

    __vector_registry = []

    class __ROLCheckpointManager__(object):
        def __init__(self):
            # directory to output checkpoints
            self.__ROL_checkpoint_dir__ = './'
            self.__ROL_mesh_file__ = None
            self.__ROL_mesh_name__ = None
            self.__index__ = 0

        def set_mesh(self, mesh_file_name, mesh_name):
            self.__ROL_mesh_file__ = mesh_file_name
            self.__ROL_mesh_name__ = mesh_name

        def set_checkpoint_dir(self, checkpoint_dir):
            # make sure we have the direcotory
            self.__makedir__(checkpoint_dir)

            self.__ROL_checkpoint_dir__ = checkpoint_dir

        def set_iteration(self, iteration):
            self.__index__ = iteration

        def __makedir__(self, dirname):
            if MPI.COMM_WORLD.rank == 0 and \
                    not os.path.isdir(dirname):
                os.mkdir(dirname)

            MPI.COMM_WORLD.Barrier()

        def increment_iteration(self):
            self.__index__ += 1

        def get_mesh_name(self):
            if None in [self.__ROL_mesh_file__, self.__ROL_mesh_name__]:
                raise ValueError(
                    "First use set_mesh to set a mesh that is checkpointable")
            return self.__ROL_mesh_file__, self.__ROL_mesh_name__

        def get_checkpoint_dir(self):

            if self.__ROL_checkpoint_dir__ is None:
                raise ValueError(
                    "set_checkpoint_dir to set the directory")

            subdir_name = os.path.join(
                self.__ROL_checkpoint_dir__,
                f"iteration_{self.__index__}")

            self.__makedir__(subdir_name)

            return subdir_name

        def get_stale_checkpoint_dir(self):
            """
                Gives the checkpoint directory
                before the current one.
                Primarily used to free up space
            """

            if self.__ROL_checkpoint_dir__ is None:
                raise ValueError(
                    "set_checkpoint_dir to set the directory")

            subdir_name = os.path.join(
                self.__ROL_checkpoint_dir__,
                f"iteration_{self.__index__ - 1}")

            if os.path.isdir(subdir_name):
                return subdir_name
            else:
                return None

    ROLCheckpointManager = __ROLCheckpointManager__()

    class CheckpointedROLVector(ROLVector):
        def __init__(self, dat, inner_product="L2"):
            super().__init__(dat, inner_product)

            with CheckpointFile(ROLCheckpointManager.get_mesh_name()[0],
                                mode='r') as fi:
                self.mesh = fi.load_mesh(ROLCheckpointManager.get_mesh_name()[1])

        def load(self):
            """Load our data once self.dat is populated"""
            with CheckpointFile(self.fname, mode='r') as ckpnt:
                for i, _ in enumerate(self.dat):
                    self.dat[i] = \
                        ckpnt.load_function(self.mesh, name=f'dat_{i}')

        def save(self, fname):
            with CheckpointFile(fname, mode='w') as ckpnt:
                ckpnt.save_mesh(self.mesh)
                for i, f in enumerate(self.dat):
                    ckpnt.save_function(f, name=f"dat_{i}")

        def clone(self):
            dat = []
            for x in self.dat:
                dat.append(x._ad_copy())
            res = CheckpointedROLVector(dat, inner_product=self.inner_product)
            res.scale(0.0)
            return res

        def __setstate__(self, state):
            """Set the state from unpickling

            Requires self.dat to be separately set, then self.load()
            can be called.
            """

            # initialise C++ state
            super().__init__(state)

            self.fname, self.inner_product = state

            with CheckpointFile(ROLCheckpointManager.get_mesh_name()[0],
                                mode='r') as fi:
                self.mesh = fi.load_mesh(ROLCheckpointManager.get_mesh_name()[1])

            __vector_registry.append(self)

        def __getstate__(self):
            """Return a state tuple suitable for pickling"""

            fname = os.path.join(
                ROLCheckpointManager.get_checkpoint_dir(),
                "vector_checkpoint_{}.h5".format(utils._new_uid()))
            self.save(fname)

            return (fname, self.inner_product)

    class CheckPointedROLSolver(ROLSolver):
        def __init__(self, problem, parameters, inner_product="L2"):
            super().__init__(problem, parameters)
            OptimizationSolver.__init__(self, problem, parameters)
            self.rolobjective = ROLObjective(problem.reduced_functional)
            x = [p.tape_value() for p in self.problem.reduced_functional.controls]
            self.rolvector = CheckpointedROLVector(x, inner_product=inner_product)
            self.params_dict = parameters

            # self.bounds = super(CheckPointedROLSolver, self).__get_bounds()
            # self.constraints = self.__get_constraints()

    class LinMoreOptimiser(object):
        def __init__(self, minimisation_problem, parameters, callback=None, checkpoint=False):

            self.callback = callback
            self.checkpoint_flag = checkpoint
            if self.checkpoint_flag:
                self.rol_solver = CheckPointedROLSolver(
                    minimisation_problem, parameters, inner_product='L2')
            else:
                self.rol_solver = ROLSolver(
                    minimisation_problem, parameters, inner_product='L2')

            self.rol_parameters = ROL.ParameterList(
                parameters, "Parameters")

            self.rol_secant = ROL.InitBFGS(
                parameters.get('General').get('Secant').get('Maximum Storage'))

            self.rol_algorithm = ROL.LinMoreAlgorithm(
                self.rol_parameters, self.rol_secant)

            self.rol_algorithm.setStatusTest(
                self.StatusTest(self.rol_parameters,
                                self.rol_solver.rolvector,
                                self),
                False)

        # solving the optimisation problem
        def run(self):
            self.rol_algorithm.run(
                self.rol_solver.rolvector,
                self.rol_solver.rolobjective,
                self.rol_solver.bounds)

        #
        def checkpoint(self):

            ROL.serialise_secant(self.rol_secant,
                                 MPI.COMM_WORLD.rank,
                                 ROLCheckpointManager.get_checkpoint_dir())

            ROL.serialise_algorithm(self.rol_algorithm,
                                    MPI.COMM_WORLD.rank,
                                    ROLCheckpointManager.get_checkpoint_dir())

            with CheckpointFile(os.path.join(
                ROLCheckpointManager.get_checkpoint_dir(),
                "solution_checkpoint.h5"),
                    mode='w') as ckpnt:
                ckpnt.save_mesh(self.rol_solver.rolvector.mesh)
                for i, f in enumerate(self.rol_solver.rolvector.dat):
                    ckpnt.save_function(f, name=f"dat_{i}")

        def reload(self, iteration):
            ROLCheckpointManager.set_iteration(iteration)

            ROL.load_secant(self.rol_secant,
                            MPI.COMM_WORLD.rank,
                            ROLCheckpointManager.get_checkpoint_dir())
            ROL.load_algorithm(self.rol_algorithm,
                               MPI.COMM_WORLD.rank,
                               ROLCheckpointManager.get_checkpoint_dir())

            # Reloading the solution
            self.rol_solver.rolvector.fname = os.path.join(
                ROLCheckpointManager.get_checkpoint_dir(),
                "solution_checkpoint.h5")
            self.rol_solver.rolvector.load()

            vec = self.rol_solver.rolvector.dat
            for v in __vector_registry:
                x = [p.copy(deepcopy=True) for p in vec]
                v.dat = x
                v.load()

        class StatusTest(ROL.StatusTest):
            def __init__(self, params, vector, parent_optimiser):
                super().__init__(params)

                # This is to access outer object
                self.parent_optimiser = parent_optimiser

                # Keep track of the vector that is being passed to StatusCheck
                self.vector = vector

                self.my_idx = 0

            @no_annotations
            def check(self, status):

                # Checkpointing
                if self.parent_optimiser.checkpoint_flag:
                    self.parent_optimiser.checkpoint()

                    # Free up space from previous checkpoint
                    if ROLCheckpointManager.get_stale_checkpoint_dir() is not None and MPI.COMM_WORLD.rank == 0:
                        shutil.rmtree(ROLCheckpointManager.get_stale_checkpoint_dir())

                    # Barriering
                    MPI.COMM_WORLD.Barrier()

                # If there is a user defined
                # callback function call it
                if self.parent_optimiser.callback is not None:
                    self.parent_optimiser.callback()

                # Write out the solution
                self.my_idx += 1
                ROLCheckpointManager.increment_iteration()

                return ROL.StatusTest.check(self, status)

except ImportError:

    class LinMoreOptimiser(object):
        def __init__(self, *args, **kwargs):
            raise ImportError("Failed to import. Please install  pyrol2.0 with checkpointing instal first")
