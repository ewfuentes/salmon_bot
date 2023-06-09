from pydrake.all import (
    AutoDiffXd,
    Context,
    Diagram,
    MathematicalProgram,
    MathematicalProgramResult,
    MultibodyPlant,
    RigidTransform,
    SnoptSolver,
    SolverOptions,
    CommonSolverOption,
    JacobianWrtVariable,
)

from typing import NamedTuple
import numpy as np


class State(NamedTuple):
    x_x: float
    z_x: float
    theta_q: float
    torso_shoulder_joint_q: float
    shoulder_hand_joint_x: float
    torso_upper_leg_joint_q: float
    upper_leg_lower_leg_joint_q: float


class StateDot(NamedTuple):
    x_v: float
    z_v: float
    theta_w: float
    torso_shoulder_joint_q: float
    shoulder_hand_joint_x: float
    torso_upper_leg_joint_q: float
    upper_leg_lower_leg_joint_q: float


class Control(NamedTuple):
    arm_force: float
    shoulder_torque: float
    hip_torque: float
    knee_torque: float


class Trajectory(NamedTuple):
    t: list[float]
    state: list[State]
    state_dot: list[StateDot]
    state_ddot: list[StateDot]
    control: list[Control]
    contact_force: np.ndarray
    is_successful: bool


INITIAL_HAND_Z = 2.11
FINAL_HAND_Z = 2.41
INITIAL_ARM_LENGTH_M = 0.4
INITIAL_HAND_X = -0.05
FINAL_HAND_X = -0.05


def get_world_from_frame(
    plant: tuple[MultibodyPlant, Context], q: State, name: str
) -> RigidTransform:
    plant[0].SetPositions(plant[1], q)
    frame = plant[0].GetFrameByName(name)
    return frame.CalcPoseInWorld(plant[1])


def frame_x_position(
    plant: tuple[MultibodyPlant, Context],
    plant_ad: tuple[MultibodyPlant, Context],
    q: np.ndarray,
    frame: str,
):
    if isinstance(q[0], AutoDiffXd):
        p = plant_ad
    else:
        p = plant
    world_from_frame = get_world_from_frame(p, q, frame)
    return world_from_frame.translation()[:1]


def frame_x_axis(
    plant: tuple[MultibodyPlant, Context],
    plant_ad: tuple[MultibodyPlant, Context],
    q: np.ndarray,
    frame: str,
):
    if isinstance(q[0], AutoDiffXd):
        p = plant_ad
    else:
        p = plant
    world_from_frame = get_world_from_frame(p, q, frame)
    return world_from_frame.rotation().multiply(np.array([1, 0, 0]))


def frame_z_position(
    plant: tuple[MultibodyPlant, Context],
    plant_ad: tuple[MultibodyPlant, Context],
    q: np.ndarray,
    frame: str,
):
    if isinstance(q[0], AutoDiffXd):
        p = plant_ad
    else:
        p = plant
    world_from_frame = get_world_from_frame(p, q, frame)
    return world_from_frame.translation()[-1:]


def add_hand_constraints(
    plant: tuple[MultibodyPlant, Context],
    plant_ad: tuple[MultibodyPlant, Context],
    q: State,
    prog: MathematicalProgram,
):
    prog.AddConstraint(
        lambda q_t: frame_z_position(plant, plant_ad, q_t, "hand"),
        lb=[FINAL_HAND_Z],
        ub=[FINAL_HAND_Z],
        vars=q[-1],
        description="hand_z_-1",
    )
    prog.AddConstraint(
        lambda q_t: frame_x_position(plant, plant_ad, q_t, "hand"),
        lb=[FINAL_HAND_X],
        ub=[FINAL_HAND_X],
        vars=q[-1],
        description="hand_x_-1",
    )

    # Don't collide with the upright
    for t in range(q.shape[0]):
        prog.AddConstraint(
            lambda q_t: frame_x_position(plant, plant_ad, q_t, "hand"),
            lb=[-np.inf],
            ub=[FINAL_HAND_X],
            vars=q[t],
            description="hand_x_-1",
        )


def add_joint_limit_constraints(
    plant: tuple[MultibodyPlant, Context],
    plant_ad: tuple[MultibodyPlant, Context],
    q: np.ndarray,
    q_dot: np.ndarray,
    u: np.ndarray,
    prog: MathematicalProgram,
):
    for t in range(q.shape[0]):
        prog.AddLinearConstraint(
            np.identity(len(q[t])),
            lb=plant[0].GetPositionLowerLimits(),
            ub=plant[0].GetPositionUpperLimits(),
            vars=q[t],
        )

        prog.AddLinearConstraint(
            np.identity(len(q_dot[t])),
            lb=plant[0].GetVelocityLowerLimits(),
            ub=plant[0].GetVelocityUpperLimits(),
            vars=q_dot[t],
        )
    for t in range(u.shape[0]):
        prog.AddLinearConstraint(
            np.identity(len(u[t])),
            lb=plant[0].GetEffortLowerLimits(),
            ub=plant[0].GetEffortUpperLimits(),
            vars=u[t],
        )


def add_time_step_constraints(
    dt: np.ndarray,
    prog: MathematicalProgram,
    min_dt_s=0.01,
    max_dt_s=0.5,
):
    for i in range(1, len(dt)):
        prog.AddLinearConstraint(
            dt[i - 1, 0] - dt[i, 0],
            lb=0.0,
            ub=0.0,
        )
    prog.AddLinearConstraint(
        dt[0, 0],
        lb=min_dt_s,
        ub=max_dt_s,
    )


def get_hand_jacobian(plant: tuple[MultibodyPlant, Context]):
    hand_frame = plant[0].GetFrameByName("hand")
    world_frame = plant[0].GetFrameByName("world")
    return plant[0].CalcJacobianTranslationalVelocity(
        plant[1],
        JacobianWrtVariable(1),
        hand_frame,
        np.zeros((3, 1)),
        world_frame,
        world_frame,
    )


def compute_manipulator_violation(
    plant: tuple[MultibodyPlant, Context],
    plant_ad: tuple[MultibodyPlant, Context],
    vars: np.ndarray,
):
    if isinstance(vars[0], AutoDiffXd):
        p = plant_ad
    else:
        p = plant
    nq = p[0].num_positions()
    nqd = p[0].num_velocities()
    nu = p[0].num_actuators()
    nf = 2
    sizes = [nq, nqd, nqd, nu, nf]
    split_at = np.cumsum(sizes)
    q, q_dot, q_ddot, u, f = np.split(vars, split_at[:-1])

    robot_idx = p[0].GetModelInstanceByName("Robot")
    p[0].SetPositions(p[1], robot_idx, q)
    p[0].SetVelocities(p[1], q_dot)
    p[0].get_actuation_input_port(robot_idx).FixValue(p[1], u)

    M = p[0].CalcMassMatrix(p[1])
    Cv = p[0].CalcBiasTerm(p[1])
    TauG = p[0].CalcGravityGeneralizedForces(p[1])

    J = get_hand_jacobian(p)
    B = p[0].MakeActuationMatrix()

    out = M @ q_ddot + Cv - TauG - J.T @ np.array([f[0], 0.0, f[1]]) - B @ u
    return out


def compute_integration_violation(
    plant: tuple[MultibodyPlant, Context],
    plant_ad: tuple[MultibodyPlant, Context],
    vars: np.ndarray,
):
    if isinstance(vars[0], AutoDiffXd):
        p = plant_ad
    else:
        p = plant
    nq = p[0].num_positions()
    nqd = p[0].num_velocities()
    sizes = [1, nq, nqd, nqd, nq, nqd, nqd]
    split_at = np.cumsum(sizes)
    dt, q_1, qdot_1, qddot_1, q_2, qdot_2, qddot_2 = np.split(vars, split_at[:-1])
    # Compute the other errors
    q_body_error = q_2 - q_1 - dt * qdot_2
    q_dot_error = qdot_2 - qdot_1 - dt * qddot_2

    error = np.concatenate([q_body_error, q_dot_error])
    return error


def add_dynamics_constraints(
    plant: tuple[MultibodyPlant, Context],
    plant_ad: tuple[MultibodyPlant, Context],
    dt: np.array,
    q: np.ndarray,
    q_dot: np.ndarray,
    q_ddot: np.ndarray,
    u: np.ndarray,
    f: np.ndarray,
    prog: MathematicalProgram,
):
    q_ddot_size = q_ddot.shape[1]
    for i in range(q.shape[0]):
        vars = np.concatenate([q[i], q_dot[i], q_ddot[i], u[i], f[i]])
        prog.AddConstraint(
            lambda vars: compute_manipulator_violation(plant, plant_ad, vars),
            lb=[0.0] * q_ddot_size,
            ub=[0.0] * q_ddot_size,
            vars=vars,
        )

    x_size = q.shape[1] + q_dot.shape[1]
    for i in range(dt.shape[0]):
        vars = np.concatenate(
            [dt[i], q[i], q_dot[i], q_ddot[i], q[i + 1], q_dot[i + 1], q_ddot[i + 1]]
        )
        prog.AddConstraint(
            lambda vars: compute_integration_violation(plant, plant_ad, vars) * 10,
            lb=[0.0] * x_size,
            ub=[0.0] * x_size,
            vars=vars,
        )


def hand_position_delta(
    plant: (MultibodyPlant, Context),
    plant_ad: (MultibodyPlant, Context),
    vars: np.ndarray,
):
    if isinstance(vars[0], AutoDiffXd):
        p = plant_ad
    else:
        p = plant
    nq = p[0].num_positions()
    sizes = [nq, nq]
    split_at = np.cumsum(sizes)
    q_0, q_t = np.split(vars, split_at[:-1])
    delta_x = frame_x_position(plant, plant_ad, q_0, "hand") - frame_x_position(
        plant, plant_ad, q_t, "hand"
    )
    delta_z = frame_z_position(plant, plant_ad, q_0, "hand") - frame_z_position(
        plant, plant_ad, q_t, "hand"
    )
    return [delta_x, delta_z]


def add_contact_constraints(
    plant: (MultibodyPlant, Context),
    plant_ad: (MultibodyPlant, Context),
    q: np.ndarray,
    f: np.ndarray,
    t_contact: int,
    prog: MathematicalProgram,
):
    for t in range(q.shape[0]):
        if t < t_contact:
            vars = np.concatenate([q[0], q[t]])
            prog.AddConstraint(
                lambda x: hand_position_delta(plant, plant_ad, x),
                lb=[0.0] * 2,
                ub=[0.0] * 2,
                vars=vars,
            )
            prog.AddBoundingBoxConstraint(
                0.0,
                np.inf,
                f[t, 1],
            )
        elif t >= t_contact:
            prog.AddBoundingBoxConstraint([0.0, 0.0], [0.0, 0.0], f[t])


def add_positive_velocity_on_release_constraint(
    plant: tuple[MultibodyPlant, Context],
    plant_ad: tuple[MultibodyPlant, Context],
    q_dot: np.ndarray,
    t_contact: int,
    prog: MathematicalProgram,
):
    state_dot = StateDot(*q_dot[t_contact])
    prog.AddBoundingBoxConstraint([0.0], [np.inf], [state_dot.z_v])


def add_initial_state_constraints(
    plant: tuple[MultibodyPlant, Context],
    plant_ad: tuple[MultibodyPlant, Context],
    q: np.ndarray,
    q_dot: np.ndarray,
    prog: MathematicalProgram,
):
    prog.AddConstraint(
        lambda q_t: frame_x_axis(plant, plant_ad, q_t, "hand"),
        lb=[-10.0, 0.0, 0.0],
        ub=[0.0, 0.0, 0.0],
        vars=q[0],
        description="hand_pointed_up_0",
    )
    prog.AddConstraint(
        lambda q_t: frame_z_position(plant, plant_ad, q_t, "hand"),
        lb=[INITIAL_HAND_Z],
        ub=[INITIAL_HAND_Z],
        vars=q[0],
        description="hand_z_0",
    )
    prog.AddConstraint(
        lambda q_t: frame_x_position(plant, plant_ad, q_t, "hand"),
        lb=[INITIAL_HAND_X],
        ub=[INITIAL_HAND_X],
        vars=q[0],
        description="hand_x_0",
    )

    prog.AddBoundingBoxConstraint(
        [INITIAL_ARM_LENGTH_M],
        [INITIAL_ARM_LENGTH_M],
        [State(*q[0]).shoulder_hand_joint_x],
    )

    prog.AddBoundingBoxConstraint(
        [np.pi - 0.01], [np.pi - 0.01], [State(*q[0]).torso_shoulder_joint_q]
    )

    prog.AddBoundingBoxConstraint(
        [0.0] * 2,
        [0.0] * 2,
        [
            State(*q[0]).torso_upper_leg_joint_q,
            State(*q[0]).upper_leg_lower_leg_joint_q,
        ],
    )

    nqd = q_dot.shape[1]
    prog.AddBoundingBoxConstraint(
        [0.0] * nqd,
        [0.0] * nqd,
        q_dot[0],
    )


def add_constraints(
    plant: tuple[MultibodyPlant, Context],
    plant_ad: tuple[MultibodyPlant, Context],
    prog: MathematicalProgram,
    t_contact: int,
    dt: np.ndarray,
    q: np.ndarray,
    q_dot: np.ndarray,
    q_ddot: np.ndarray,
    u: np.ndarray,
    f: np.ndarray,
):
    add_initial_state_constraints(plant, plant_ad, q, q_dot, prog)
    add_contact_constraints(plant, plant_ad, q, f, t_contact, prog)
    add_hand_constraints(plant, plant_ad, q, prog)
    add_joint_limit_constraints(plant, plant_ad, q, q_dot, u, prog)
    add_positive_velocity_on_release_constraint(plant, plant_ad, q_dot, t_contact, prog)
    add_time_step_constraints(dt, t_contact, prog)
    add_dynamics_constraints(plant, plant_ad, dt, q, q_dot, q_ddot, u, f, prog)


def get_key_frames(plant: tuple[MultibodyPlant, Context]):
    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(rows=3, cols=plant[0].num_positions())

    # Add hand z constraints
    prog.AddConstraint(
        lambda q_t: frame_z_position(plant, plant, q_t, "hand"),
        lb=[INITIAL_HAND_Z],
        ub=[INITIAL_HAND_Z],
        vars=q[0],
    )
    prog.AddConstraint(
        lambda q_t: frame_z_position(plant, plant, q_t, "hand"),
        lb=[INITIAL_HAND_Z],
        ub=[INITIAL_HAND_Z],
        vars=q[1],
    )
    prog.AddConstraint(
        lambda q_t: frame_z_position(plant, plant, q_t, "hand"),
        lb=[FINAL_HAND_Z],
        ub=[FINAL_HAND_Z],
        vars=q[2],
    )

    prog.AddConstraint(
        lambda q_t: frame_x_axis(plant, plant, q_t, "hand"),
        lb=[0, 0, 0],
        ub=[0, 0, 10],
        vars=q[1],
    )

    # Add hand x constraints
    for i in range(3):
        prog.AddConstraint(
            lambda q_t: frame_x_position(plant, plant, q_t, "hand"),
            lb=[INITIAL_HAND_X],
            ub=[INITIAL_HAND_X],
            vars=q[i],
        )

    # Add arm angle constraints
    prog.AddBoundingBoxConstraint(
        [np.pi],
        [np.pi],
        [State(*q[0]).torso_shoulder_joint_q],
    )
    prog.AddBoundingBoxConstraint(
        [np.pi / 2],
        [np.pi / 2],
        [State(*q[1]).torso_shoulder_joint_q],
    )
    prog.AddBoundingBoxConstraint(
        [np.pi],
        [np.pi],
        [State(*q[2]).torso_shoulder_joint_q],
    )

    state_guess = np.array(
        [
            State(
                x_x=0.0,
                z_x=0.0,
                theta_q=0.0,
                torso_shoulder_joint_q=np.pi,
                shoulder_hand_joint_x=INITIAL_ARM_LENGTH_M,
                torso_upper_leg_joint_q=np.pi / 2.0,
                upper_leg_lower_leg_joint_q=np.pi / 2.0,
            ),
            State(
                x_x=0.0,
                z_x=0.0,
                theta_q=0.0,
                torso_shoulder_joint_q=np.pi / 2.0,
                shoulder_hand_joint_x=0.2,
                torso_upper_leg_joint_q=np.pi / 2.0,
                upper_leg_lower_leg_joint_q=np.pi / 2.0,
            ),
            State(
                x_x=0.0,
                z_x=0.0,
                theta_q=0.0,
                torso_shoulder_joint_q=np.pi,
                shoulder_hand_joint_x=0.4,
                torso_upper_leg_joint_q=np.pi / 2.0,
                upper_leg_lower_leg_joint_q=np.pi / 2.0,
            ),
        ]
    )

    initial_guess = np.zeros(prog.num_vars())
    prog.SetDecisionVariableValueInVector(q, state_guess, initial_guess)

    solver = SnoptSolver()
    result = solver.Solve(prog, initial_guess)
    print("key frame opt result:", result.get_solution_result())
    return result.GetSolution(q)


def get_initial_guess(
    num_timesteps: int,
    t_contact: int,
    prog: MathematicalProgram,
    q: np.ndarray,
    f: np.ndarray,
    plant: tuple[MultibodyPlant, Context],
) -> np.ndarray:
    key_frames = get_key_frames(plant)
    out = np.ones(prog.num_vars())
    q_guess = np.zeros_like(q)

    t_flight = q.shape[0] - t_contact

    def lerp(a: State, b: State, frac: float):
        return State(*[(b[i] - a[i]) * frac + a[i] for i in range(len(a))])

    initial_state = State(*key_frames[0])
    release_state = State(*key_frames[1])
    catch_state = State(*key_frames[2])

    for t in range(t_contact):
        q_guess[t, :] = lerp(initial_state, release_state, t / t_contact)
    for t in range(t_flight):
        q_guess[t + t_contact, :] = lerp(release_state, catch_state, t / t_flight)
    prog.SetDecisionVariableValueInVector(q, q_guess, out)
    return out


def package_trajectory(
    prog: MathematicalProgram,
    result: MathematicalProgramResult,
    dt: np.ndarray,
    q: np.ndarray,
    q_dot: np.ndarray,
    q_ddot: np.ndarray,
    u: np.ndarray,
    f: np.ndarray,
):
    print(result.get_solution_result())
    return Trajectory(
        t=result.GetSolution(dt),
        state=result.GetSolution(q),
        state_dot=result.GetSolution(q_dot),
        state_ddot=result.GetSolution(q_ddot),
        control=result.GetSolution(u),
        contact_force=result.GetSolution(f),
        is_successful=result.is_success(),
    )


def add_costs(
    plant: tuple[MultibodyPlant, Context],
    plant_ad: tuple[MultibodyPlant, Context],
    prog: MathematicalProgram,
    q: np.ndarray,
    q_dot: np.ndarray,
    q_ddot: np.ndarray,
    u: np.ndarray,
):
    nq = q.shape[1]
    nu = u.shape[1]
    for t in range(q.shape[0]):
        prog.AddQuadraticCost(
            Q=np.identity(nq) * 1.0, b=np.zeros((nq)), vars=q[t], is_convex=True
        )
        prog.AddQuadraticCost(
            Q=np.identity(nq) * 10.0, b=np.zeros((nq)), vars=q_dot[t], is_convex=True
        )
        prog.AddQuadraticCost(
            Q=np.identity(nq) * 1.0, b=np.zeros((nq)), vars=q_ddot[t], is_convex=True
        )

    for t in range(u.shape[0]):
        prog.AddQuadraticCost(
            Q=np.diag([10.0, 10.0, 1.0, 1.0]),
            b=np.zeros((nu,)),
            vars=u[t],
            is_convex=True,
        )
