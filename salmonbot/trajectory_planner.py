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
                description=f"{frame}_in_plane_{t}",
            )


def add_hand_height_constraints(
    plant: MultibodyPlant, q: State, prog: MathematicalProgram
):
    prog.AddConstraint(
        lambda q_t: frame_z_position(plant, q_t, "hand"),
        lb=[2.0],
        ub=[2.0],
        vars=q[0],
        description="hand_height_0",
    )
    prog.AddConstraint(
        lambda q_t: frame_z_position(plant, q_t, "hand"),
        lb=[3.0],
        ub=[3.0],
        vars=q[-1],
        description="hand_height_-1",
    )


def add_joint_limit_constraints(
    plant: MultibodyPlant, q: np.ndarray, q_dot: np.ndarray, prog: MathematicalProgram
):
    for t in range(q.shape[0]):
        prog.AddLinearConstraint(
            np.identity(len(q[t])),
            lb=plant.GetPositionLowerLimits(),
            ub=plant.GetPositionUpperLimits(),
            vars=q[t],
        )


def add_periodicity_constraints(plant, q, q_dot, prog):
    ndim = q_dot.shape[1]
    prog.AddLinearConstraint(
        q_dot[-1] - q_dot[0],
        lb=np.zeros((ndim,)),
        ub=np.zeros((ndim,)),
    )

    q_delta = np.array(
        State(
            world_torso_qw=0.0,
            world_torso_qx=0.0,
            world_torso_qy=0.0,
            world_torso_qz=0.0,
            world_torso_x=0.0,
            world_torso_y=0.0,
            world_torso_z=np.inf,
            torso_shoulder_joint_q=0.0,
            shoulder_hand_joint_x=0.0,
            torso_upper_leg_joint_q=0.0,
            upper_leg_lower_leg_joint_q=0.0,
        )
    )

    prog.AddLinearConstraint(
        q[-1] - q[0],
        lb=-q_delta,
        ub=q_delta,
    )


def add_time_step_constraints(
    dt: np.ndarray, prog: MathematicalProgram, min_dt_s=0.005, max_dt_s=0.1
):
    prog.AddLinearConstraint(
        dt, lb=np.ones_like(dt) * min_dt_s, ub=np.ones_like(dt) * max_dt_s
    )


def compute_manipulator_violation(plant: MultibodyPlant, vars):
    nq = plant.num_positions()
    nqd = plant.num_velocities()
    nu = plant.num_actuators()
    sizes = [nq, nqd, nqd, nu]
    split_at = np.cumsum(sizes)
    q, q_dot, q_ddot, u = np.split(vars, split_at[:-1])

    context = plant.CreateDefaultContext()
    plant.SetPositions(context, q)
    plant.SetVelocities(context, q_dot)

    M = plant.CalcMassMatrix(context)
    Cv = plant.CalcBiasTerm(context)
    TauG = plant.CalcGravityGeneralizedForces(context)

    # TODO: Add contact force
    # TODO: Add actuation
    return M @ q_ddot + Cv - TauG


def compute_integration_violation(plant, vars):
    nq = plant.num_positions()
    nqd = plant.num_velocities()
    nu = plant.num_actuators()
    sizes = [1, nq, nq, nqd, nqd, nqd, nu]
    split_at = np.cumsum(sizes)
    dt, q_1, q_2, qdot_1, qdot_2, qddot_2, u = np.split(vars, split_at[:-1])

    print(q_1.shape, qdot_1.shape, qddot_2.shape)
    x_1 = np.concatenate([q_1, qdot_1])
    x_2 = np.concatenate([q_2, qdot_2])
    xdot_2 = np.concatenate([qdot_2, qddot_2])

    return x_2 - x_1 - dt * xdot_2


def add_dynamics_constraints(
    plant: MultibodyPlant,
    dt: np.array,
    q: np.ndarray,
    q_dot: np.ndarray,
    q_ddot: np.ndarray,
    u: np.ndarray,
    prog: MathematicalProgram,
):

    q_ddot_size = q_ddot.shape[1]
    for i in range(u.shape[0]):
        vars = np.concatenate([q[i], q_dot[i], q_ddot[i], u[i]])
        prog.AddConstraint(
            lambda vars: compute_manipulator_violation(plant, vars),
            lb=[0.0] * q_ddot_size,
            ub=[0.0] * q_ddot_size,
            vars=vars,
        )
    for i in range(dt.shape[0]):
        vars = np.concatenate([q[i], q_dot[i], q_ddot[i], q[i+1], q_dot[i+1], q_ddot[i+1]])
        prog.AddConstraint(
            lambda vars: compute_integration_violation(plant, vars),
            lb=[0.0] * x_size,
            ub=[0.0] * x_size,
            vars=vars,
        )


def add_constraints(
    plant: MultibodyPlant,
    prog: MathematicalProgram,
    dt: np.ndarray,
    q: np.ndarray,
    q_dot: np.ndarray,
    q_ddot: np.ndarray,
    u: np.ndarray,
):
    add_plane_constraints(plant, q, prog)
    add_hand_height_constraints(plant, q, prog)
    add_joint_limit_constraints(plant, q, q_dot, prog)
    add_periodicity_constraints(plant, q, q_dot, prog)
    add_time_step_constraints(dt, prog)
    add_dynamics_constraints(plant, dt, q, q_dot, q_ddot, u, prog)


def get_initial_guess(
    num_timesteps: int, prog: MathematicalProgram, q: np.ndarray
) -> np.ndarray:
    out = np.ones(prog.num_vars())
    q_guess = np.zeros_like(q)
    for t in range(q.shape[0]):
        q_guess[t, :] = State(1, *([0.0] * (len(State._fields) - 1)))
    prog.SetDecisionVariableValueInVector(q, q_guess, out)
    return out


def plan_trajectory(plant: MultibodyPlant):
    T = 50
    plant_ad = plant.ToAutoDiffXd()
    prog = MathematicalProgram()

    nq = plant.num_positions()
    nqd = plant.num_velocities()
    nu = plant.num_actuators()

    dt = prog.NewContinuousVariables(rows=T, cols=1, name="dt")
    q = prog.NewContinuousVariables(rows=T + 1, cols=nq, name="q")
    q_dot = prog.NewContinuousVariables(rows=T + 1, cols=nqd, name="q_dot")
    q_ddot = prog.NewContinuousVariables(rows=T + 1, cols=nqd, name="q_ddot")
    u = prog.NewContinuousVariables(rows=T, cols=nu, name="u")

    add_constraints(plant_ad, prog, dt, q, q_dot, q_ddot, u)

    initial_guess = get_initial_guess(T, prog, q)

    solver = SnoptSolver()
    result = solver.Solve(prog, initial_guess=initial_guess)
    print(f"Optimization Succeeded? {result.is_success()}")
    IPython.embed()
