from pydrake.all import (
    MathematicalProgram,
    MultibodyPlant,
    RigidTransform,
    SnoptSolver,
)

from typing import NamedTuple
import IPython
import numpy as np


class State(NamedTuple):
    world_torso_qw: float
    world_torso_qx: float
    world_torso_qy: float
    world_torso_qz: float
    world_torso_x: float
    world_torso_y: float
    world_torso_z: float
    torso_shoulder_joint_q: float
    shoulder_hand_joint_x: float
    torso_upper_leg_joint_q: float
    upper_leg_lower_leg_joint_q: float


def get_world_from_frame(plant: MultibodyPlant, q: State, name: str) -> RigidTransform:
    context = plant.CreateDefaultContext()
    plant.SetPositions(context, q)
    hand_frame = plant.GetFrameByName(name)
    return hand_frame.CalcPoseInWorld(context)


def frame_y_position(plant: MultibodyPlant, q: np.ndarray, frame: str):
    world_from_frame = get_world_from_frame(plant, q, frame)
    return world_from_frame.translation()[1:2]


def frame_z_position(plant: MultibodyPlant, q: np.ndarray, frame: str):
    # print(type(q[0]))
    world_from_frame = get_world_from_frame(plant, q, frame)
    return world_from_frame.translation()[-1:]


def add_plane_constraints(
    plant: MultibodyPlant, q: np.ndarray, prog: MathematicalProgram
):
    for t in range(q.shape[0]):
        for frame in ["hand", "torso", "lower_leg"]:
            prog.AddConstraint(
                lambda q_t: frame_y_position(plant, q_t, frame),
                lb=[0.0],
                ub=[0.0],
                vars=q[t],
                description=f'{frame}_in_plane_{t}'
            )


def add_hand_height_constraints(
    plant: MultibodyPlant, q: State, prog: MathematicalProgram
):
    prog.AddConstraint(
        lambda q_t: frame_z_position(plant, q_t, "hand"), lb=[2.0], ub=[2.0], vars=q[0], description='hand_height_0'
    )
    prog.AddConstraint(
        lambda q_t: frame_z_position(plant, q_t, "hand"), lb=[3.0], ub=[3.0], vars=q[-1], description='hand_height_-1'
    )


def add_constraints(
    plant: MultibodyPlant,
    prog: MathematicalProgram,
    q: np.ndarray,
    q_dot: np.ndarray,
):
    add_plane_constraints(plant, q, prog)
    add_hand_height_constraints(plant, q, prog)


def get_initial_guess(num_timesteps: int, prog: MathematicalProgram,
                      q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
    out = np.ones(prog.num_vars())
    q_guess = np.zeros_like(q)
    for t in range(q.shape[0]):
        q_guess[t, :] = State(1, *([0.0] * (len(State._fields) - 1)))
    prog.SetDecisionVariableValueInVector(q, q_guess, out)
    return out


def plan_trajectory(plant: MultibodyPlant):
    plant_ad = plant.ToAutoDiffXd()
    prog = MathematicalProgram()

    nq = plant.num_positions()

    T = 50
    q = prog.NewContinuousVariables(rows=T + 1, cols=nq, name="q")
    q_dot = prog.NewContinuousVariables(rows=T + 1, cols=nq, name="q_dot")

    add_constraints(plant_ad, prog, q, q_dot)

    initial_guess = get_initial_guess(T, prog, q, q_dot)

    solver = SnoptSolver()
    result = solver.Solve(prog, initial_guess=initial_guess)
    print(f'Optimization Succeeded? {result.is_success()}')
    IPython.embed()
