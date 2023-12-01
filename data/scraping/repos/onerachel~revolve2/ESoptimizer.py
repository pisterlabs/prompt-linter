import math
from random import Random
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from pyrr import Quaternion, Vector3
from revolve2.actor_controllers.cpg import CpgNetworkStructure
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import Body, ModularRobot
from revolve2.core.modular_robot.brains import (
    BrainCpgNetworkStatic,
    make_cpg_network_structure_neighbour,
)
from revolve2.core.optimization import ProcessIdGen
from revolve2.core.optimization.ea.openai_es import OpenaiESOptimizer
from revolve2.core.physics.actor import Actor
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    Environment,
    PosedActor,
    Runner,
)
from revolve2.runners.isaacgym import LocalRunner


class ESOptimizer(OpenaiESOptimizer):
    _body: Body
    _actor: Actor
    _dof_ids: List[int]
    _cpg_network_structure: CpgNetworkStructure

    _runner: Runner
    _controllers: List[ActorController]

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _num_generations: int

    async def ainit_new(
            # type: ignore # TODO for now ignoring mypy complaint about LSP problem, override parent's ainit
            self,
            database: AsyncEngine,
            session: AsyncSession,
            process_id: int,
            process_id_gen: ProcessIdGen,
            rng: Random,
            population_size: int,
            sigma: float,
            learning_rate: float,
            robot_body: Body,
            initial_mean: npt.NDArray[np.float_],
            simulation_time: int,
            sampling_frequency: float,
            control_frequency: float,
            num_generations: int,
            cpg_structure: CpgNetworkStructure,
    ) -> None:
        self._actor, self._dof_ids = robot_body.to_actor()
        self._cpg_network_structure = cpg_structure

        self._body = robot_body # ???

        # nprng = np.random.Generator(
        #     np.random.PCG64(rng.randint(0, 2**63))
        # )  # rng is currently not numpy, but this would be very convenient. do this until that is resolved.
        # initial_mean = nprng.standard_normal(self._cpg_network_structure.num_params)

        await super().ainit_new(
            database=database,
            session=session,
            process_id=process_id,
            process_id_gen=process_id_gen,
            rng=rng,
            population_size=population_size,
            sigma=sigma,
            learning_rate=learning_rate,
            initial_mean=initial_mean,
        )

        self._init_runner()

        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations

    async def ainit_from_database(  # type: ignore # see comment at ainit_new
            self,
            database: AsyncEngine,
            session: AsyncSession,
            process_id: int,
            process_id_gen: ProcessIdGen,
            rng: Random,
            robot_body: Body,
            simulation_time: int,
            sampling_frequency: float,
            control_frequency: float,
            num_generations: int,
            cpg_structure: CpgNetworkStructure,
    ) -> bool:
        if not await super().ainit_from_database(
                database=database,
                session=session,
                process_id=process_id,
                process_id_gen=process_id_gen,
                rng=rng,
        ):
            return False

        self._body = robot_body # ???
        self._actor, self._dof_ids = robot_body.to_actor()
        self._cpg_network_structure = cpg_structure

        self._init_runner()

        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations

        return True

    def _init_runner(self) -> None:
        self._runner = LocalRunner(LocalRunner.SimParams(), headless=True)

    async def _evaluate_population(
            self,
            database: AsyncEngine,
            process_id: int,
            process_id_gen: ProcessIdGen,
            population: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
            control=self._control,
        )

        self._controllers = []

        for params in population:
            initial_state = self._cpg_network_structure.make_uniform_state(
                0.5 * math.pi / 2.0
            )
            weight_matrix = self._cpg_network_structure.make_weight_matrix_from_params(params)
            dof_ranges = self._cpg_network_structure.make_uniform_dof_ranges(1.0)
            brain = BrainCpgNetworkStatic(
                initial_state,
                self._cpg_network_structure.num_cpgs,
                weight_matrix,
                dof_ranges,
            )
            controller = brain.make_controller(self._body, self._dof_ids)

            bounding_box = self._actor.calc_aabb()
            self._controllers.append(controller)
            env = Environment()
            env.actors.append(
                PosedActor(
                    self._actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0 - bounding_box.offset.z,
                        ]
                    ),
                    Quaternion(),
                    [0.0 for _ in controller.get_dof_targets()],
                )
            )
            batch.environments.append(env)

        states = await self._runner.run_batch(batch)

        return np.array(
            [
                self._calculate_fitness(
                    states[0].envs[i].actor_states[0],
                    states[-1].envs[i].actor_states[0],
                )
                for i in range(len(population))
            ]
        )

    def _control(self, dt: float, control: ActorControl) -> None:
        for control_i, controller in enumerate(self._controllers):
            controller.step(dt)
            control.set_dof_targets(control_i, 0, controller.get_dof_targets())

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState) -> float:
        # TODO simulation can continue slightly passed the defined sim time.

        # distance traveled on the xy plane
        return math.sqrt(
            (begin_state.position[0] - end_state.position[0]) ** 2
            + ((begin_state.position[1] - end_state.position[1]) ** 2)
        )

    def _must_do_next_gen(self) -> bool:
        return self.generation_number != self._num_generations
