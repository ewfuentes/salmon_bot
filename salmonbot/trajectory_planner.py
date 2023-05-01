from pydrake.all import (
    AutoDiffXd,
    Context,
    Diagram,
    MathematicalProgram,
    MathematicalProgramResult,
    MultibodyPlant,
    RigidTransform,
    SnoptSolver,
    QuaternionEulerIntegrationConstraint,
    UnitQuaternionConstraint,
    SolverOptions,
    CommonSolverOption,
    JacobianWrtVariable,
)

from typing import NamedTuple
import numpy as np
import IPython


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


class StateDot(NamedTuple):
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


def get_world_from_frame(
    plant: tuple[MultibodyPlant, Context], q: State, name: str
) -> RigidTransform:
    plant[0].SetPositions(plant[1], q)
    frame = plant[0].GetFrameByName(name)
    return frame.CalcPoseInWorld(plant[1])


def xz_plane_constraint(
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

    frame_origin_in_world = world_from_frame.translation()
    frame_x_in_world = (
        world_from_frame.multiply(np.array([[1.0, 0.0, 0.0]]).T).squeeze().tolist()
    )
    frame_z_in_world = (
        world_from_frame.multiply(np.array([[0.0, 0.0, 1.0]]).T).squeeze().tolist()
    )

    return [frame_origin_in_world[1], frame_x_in_world[1], frame_z_in_world[1]]


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


def add_plane_constraints(
    plant: tuple[MultibodyPlant, Context],
    plant_ad: tuple[MultibodyPlant, Context],
    q: np.ndarray,
    q_dot: np.ndarray,
    prog: MathematicalProgram,
):
    for t in range(q.shape[0]):
        for frame in ["hand", "torso", "lower_leg"]:
            prog.AddConstraint(
                lambda q_t: xz_plane_constraint(plant, plant_ad, q_t, frame),
                lb=[0.0] * 3,
                ub=[0.0] * 3,
                vars=q[t],
                description=f"{frame}_in_plane_{t}",
            )


def add_hand_height_constraints(
    plant: tuple[MultibodyPlant, Context],
    plant_ad: tuple[MultibodyPlant, Context],
    q: State,
    prog: MathematicalProgram,
):
    prog.AddConstraint(
        lambda q_t: frame_z_position(plant, plant_ad, q_t, "hand"),
        lb=[2.11],
        ub=[2.11],
        vars=q[0],
        description="hand_height_0",
    )
    prog.AddConstraint(
        lambda q_t: frame_z_position(plant, plant_ad, q_t, "hand"),
        lb=[2.41],
        ub=[2.41],
        vars=q[-1],
        description="hand_height_-1",
    )
    prog.AddConstraint(
        lambda q_t: frame_x_position(plant, plant_ad, q_t, "hand"),
        lb=[-0.05],
        ub=[-0.05],
        vars=q[0],
        description="hand_height_0",
    )
    prog.AddConstraint(
        lambda q_t: frame_x_position(plant, plant_ad, q_t, "hand"),
        lb=[-0.05],
        ub=[-0.05],
        vars=q[-1],
        description="hand_height_-1",
    )


def add_joint_limit_constraints(
    plant: tuple[MultibodyPlant, Context],
    plant_ad: tuple[MultibodyPlant, Context],
    q: np.ndarray,
    q_dot: np.ndarray,
    prog: MathematicalProgram,
):
    for t in range(q.shape[0]):
        prog.AddLinearConstraint(
            np.identity(len(q[t])),
            lb=plant[0].GetPositionLowerLimits(),
            ub=plant[0].GetPositionUpperLimits(),
            vars=q[t],
        )


def add_time_step_constraints(
    dt: np.ndarray, prog: MathematicalProgram, min_dt_s=0.0005, max_dt_s=0.05
):
    prog.AddLinearConstraint(
        dt, lb=np.ones_like(dt) * min_dt_s, ub=np.ones_like(dt) * max_dt_s
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

    out = M @ q_ddot + Cv - TauG - J.T @ np.array([f[0], 0.0, f[1]]) + B @ u
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
    q_body_error = q_2[4:] - q_1[4:] - dt * qdot_2[3:]
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
    quaternion_constraint = QuaternionEulerIntegrationConstraint(
        allow_quaternion_negation=False
    )
    for i in range(dt.shape[0]):
        prog.AddConstraint(
            quaternion_constraint,
            np.concatenate([q[i, :4], q[i + 1, :4], q_dot[i + 1, :3], dt[i]]),
        )

    for i in range(q.shape[0]):
        prog.AddConstraint(UnitQuaternionConstraint(), q[i, :4])

    q_ddot_size = q_ddot.shape[1]
    for i in range(u.shape[0]):
        vars = np.concatenate([q[i], q_dot[i], q_ddot[i], u[i], f[i]])
        prog.AddConstraint(
            lambda vars: compute_manipulator_violation(plant, plant_ad, vars),
            lb=[0.0] * q_ddot_size,
            ub=[0.0] * q_ddot_size,
            vars=vars,
        )

    x_size = q.shape[1] - 4 + q_dot.shape[1]
    for i in range(dt.shape[0]):
        vars = np.concatenate(
            [dt[i], q[i], q_dot[i], q_ddot[i], q[i + 1], q_dot[i + 1], q_ddot[i + 1]]
        )
        prog.AddConstraint(
            lambda vars: compute_integration_violation(plant, plant_ad, vars),
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
        if t <= t_contact:
            vars = np.concatenate([q[0], q[t]])
            prog.AddConstraint(
                lambda x: hand_position_delta(plant, plant_ad, x),
                lb=[0.0] * 2,
                ub=[0.0] * 2,
                vars=vars,
            )
        elif t > t_contact:
            prog.AddBoundingBoxConstraint([0.0, 0.0], [0.0, 0.0], f[t])


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
    add_contact_constraints(plant, plant_ad, q, f, t_contact, prog)
    add_plane_constraints(plant, plant_ad, q, q_dot, prog)
    add_hand_height_constraints(plant, plant_ad, q, prog)
    add_joint_limit_constraints(plant, plant_ad, q, q_dot, prog)
    add_time_step_constraints(dt, prog)
    add_dynamics_constraints(plant, plant_ad, dt, q, q_dot, q_ddot, u, f, prog)


def get_initial_guess(
    num_timesteps: int,
    t_contact: int,
    prog: MathematicalProgram,
    q: np.ndarray,
    q_dot: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    out = np.ones(prog.num_vars())
    q_guess = np.zeros_like(q)
    q_dot_guess = np.zeros_like(q_dot)
    initial_z = 2.11
    final_z = 2.11 + 0.3
    z_step = (final_z - initial_z) / (num_timesteps - t_contact)
    for t in range(q.shape[0]):
        dz = z_step * t if t > t_contact else 0.0
        z_target = initial_z + dz
        q_guess[t, :] = State(
            world_torso_qw=1.0,
            world_torso_qx=0.0,
            world_torso_qy=0.0,
            world_torso_qz=0.0,
            world_torso_x=-0.05,
            world_torso_y=0.0,
            world_torso_z=z_target,
            torso_shoulder_joint_q=np.pi,
            shoulder_hand_joint_x=0.3,
            torso_upper_leg_joint_q=0.0,
            upper_leg_lower_leg_joint_q=0.0,
        )

        q_dot_guess[t, :] = StateDot(
            world_torso_qx=0.0,
            world_torso_qy=0.1,
            world_torso_qz=0.0,
            world_torso_x=0.0,
            world_torso_y=0.0,
            world_torso_z=dz,
            torso_shoulder_joint_q=0.0,
            shoulder_hand_joint_x=0.0,
            torso_upper_leg_joint_q=0.0,
            upper_leg_lower_leg_joint_q=0.0,
        )
    prog.SetDecisionVariableValueInVector(q, q_guess, out)
    prog.SetDecisionVariableValueInVector(q_dot, q_dot_guess, out)
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


def plan_trajectory(diagram: Diagram) -> Trajectory:
    T = 100
    T_CONTACT = 50
    root_context = diagram.CreateDefaultContext()
    plant = diagram.GetSubsystemByName("plant")
    context = plant.GetMyContextFromRoot(root_context)

    diagram_ad = diagram.ToAutoDiffXd()
    root_context_ad = diagram_ad.CreateDefaultContext()
    plant_ad = diagram_ad.GetSubsystemByName("plant")
    context_ad = plant_ad.GetMyContextFromRoot(root_context_ad)
    prog = MathematicalProgram()

    nq = plant_ad.num_positions()
    nqd = plant_ad.num_velocities()
    nu = plant_ad.num_actuators()

    dt = prog.NewContinuousVariables(rows=T, cols=1, name="dt")
    q = prog.NewContinuousVariables(rows=T + 1, cols=nq, name="q")
    q_dot = prog.NewContinuousVariables(rows=T + 1, cols=nqd, name="q_dot")
    q_ddot = prog.NewContinuousVariables(rows=T + 1, cols=nqd, name="q_ddot")
    u = prog.NewContinuousVariables(rows=T, cols=nu, name="u")
    f = prog.NewContinuousVariables(rows=T + 1, cols=2, name="f")

    add_constraints(
        (plant, context),
        (plant_ad, context_ad),
        prog,
        T_CONTACT,
        dt,
        q,
        q_dot,
        q_ddot,
        u,
        f,
    )

    for t in range(u.shape[0]):
        prog.AddL2NormCost(A=np.diag([0.1, 0.1, 1.0, 1.0]), b=np.zeros((4, 1)), vars=u[t])

    initial_guess = get_initial_guess(T, T_CONTACT, prog, q, q_dot, f)

    solver = SnoptSolver()
    solver_options = SolverOptions()
    solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    # solver_options.SetOption(SnoptSolver.id(), "Iterations limit", 1000000)
    # solver_options.SetOption(SnoptSolver.id(), "Major Iterations limit", 1000000)
    print(solver_options.GetOptions(solver.id()))
    result = solver.Solve(
        prog, initial_guess=initial_guess, solver_options=solver_options
    )
    return package_trajectory(prog, result, dt, q, q_dot, q_ddot, u, f)
