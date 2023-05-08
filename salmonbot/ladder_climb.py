import IPython
import numpy as np

from pydrake.all import (
    Diagram,
    MathematicalProgram,
    MathematicalProgramResult,
    SnoptSolver,
    MultibodyPlant,
    Context,
    CommonSolverOption,
    SolverOptions,
)

from salmonbot.trajectory_planner import (
    Trajectory,
    State,
    frame_x_position,
    frame_x_axis,
    frame_z_position,
    add_contact_constraints,
    add_joint_limit_constraints,
    add_time_step_constraints,
    add_dynamics_constraints,
    add_positive_velocity_on_release_constraint,
)


def get_initial_state(
    prog: MathematicalProgram,
    pc: tuple[MultibodyPlant, Context],
    pc_ad: tuple[MultibodyPlant, Context],
    target_x_m: float,
    target_z_m: float,
) -> Trajectory:
    nq = pc[0].num_positions()
    nqd = pc[0].num_velocities()

    q = prog.NewContinuousVariables(cols=nq, rows=1, name="q_init")
    q_dot = prog.NewContinuousVariables(cols=nqd, rows=1, name="q_dot_init")
    q_ddot = prog.NewContinuousVariables(cols=nqd, rows=1, name="q_ddot_init")

    s = State(*q[0])
    # Zero Velocities and derivatives
    prog.AddBoundingBoxConstraint(np.zeros_like(q_dot), np.zeros_like(q_dot), q_dot)
    prog.AddBoundingBoxConstraint(np.zeros_like(q_dot), np.zeros_like(q_dot), q_ddot)

    # Legs Straight, body upright, arms almost straight up
    eps = 1e-3
    prog.AddBoundingBoxConstraint(
        [0.0, 0.0, np.pi - eps, 0.4],
        [0.0, 0.0, np.pi - eps, 0.4],
        [
            s.torso_upper_leg_joint_q,
            s.upper_leg_lower_leg_joint_q,
            s.torso_shoulder_joint_q,
            s.shoulder_hand_joint_x,
        ],
    )

    # Ensure that we start at the appropriate location
    prog.AddConstraint(
        lambda q_t: frame_z_position(pc, pc_ad, q_t, "hand"),
        lb=[target_z_m],
        ub=[target_z_m],
        vars=q[0],
        description="hand_z_0",
    )
    prog.AddConstraint(
        lambda q_t: frame_x_position(pc, pc_ad, q_t, "hand"),
        lb=[target_x_m],
        ub=[target_x_m],
        vars=q[0],
        description="hand_x_0",
    )
    # prog.AddConstraint(
    #     lambda q_t: frame_x_axis(pc, pc_ad, q_t, "hand"),
    #     lb=[-10.0, 0.0, 0.0],
    #     ub=[0.0, 0.0, 0.0],
    #     vars=q[0],
    #     description="hand_pointed_up_0",
    # )

    return Trajectory(
        t=None,
        state=q,
        state_dot=q_dot,
        state_ddot=q_ddot,
        control=None,
        contact_force=None,
        is_successful=False,
    )


def add_cycle_consistency_constraint(
    prog: MathematicalProgram, swing_up_traj: Trajectory, flight_traj: Trajectory
):
    s0 = State(*swing_up_traj.state[0])
    s1 = State(*flight_traj.state[-1])
    for i, field_name in enumerate(State._fields):
        if field_name == "z_x":
            continue
        prog.AddConstraint(s0[i] - s1[i] == 0.0)

    for i in range(swing_up_traj.state_dot.shape[1]):
        prog.AddConstraint(
            swing_up_traj.state_dot[0, i] - flight_traj.state_dot[-1, i] == 0.0
        )
        prog.AddConstraint(
            swing_up_traj.state_ddot[0, i] - flight_traj.state_ddot[-1, i] == 0.0
        )


def add_consistency_constraint(
    prog: MathematicalProgram,
    prev_traj: Trajectory,
    new_traj: Trajectory,
):
    nq = prev_traj.state.shape[1]
    for i in range(nq):
        prog.AddConstraint(prev_traj.state[-1, i] - new_traj.state[0, i] == 0.0)
        prog.AddConstraint(prev_traj.state_dot[-1, i] - new_traj.state_dot[0, i] == 0.0)
        prog.AddConstraint(
            prev_traj.state_ddot[-1, i] - new_traj.state_ddot[0, i] == 0.0
        )


def add_trajectory_dynamics_constraints(
    pc: tuple[MultibodyPlant, Context],
    pc_ad: tuple[MultibodyPlant, Context],
    traj: Trajectory,
    prog: MathematicalProgram,
):
    add_dynamics_constraints(
        pc,
        pc_ad,
        traj.t,
        traj.state,
        traj.state_dot,
        traj.state_ddot,
        traj.control,
        traj.contact_force,
        prog,
    )


def add_trajectory_costs(
    traj: Trajectory,
    prog: MathematicalProgram,
):
    dt = traj.t[0]
    nqd = traj.state_dot.shape[1]
    nu = traj.control.shape[1]

    def control_cost(vars: np.ndarray):
        ndt = 1
        sizes = [ndt, nu]
        split_at = np.cumsum(sizes)
        dt, u = np.split(vars, split_at[:-1])

        return dt[0] * 100.0 * np.sum(u @ np.diag([0.1, 0.1, 1.0, 1.0]) @ u)

    def state_cost(vars: np.ndarray):
        ndt = 1
        sizes = [ndt, nqd]
        split_at = np.cumsum(sizes)
        dt, q_ddot = np.split(vars, split_at[:-1])

        return dt[0] * 100.0 * np.sum(q_ddot * q_ddot)

    for t in range(traj.state.shape[0]):
        prog.AddCost(control_cost, np.concatenate([dt, traj.control[t]]))
        prog.AddCost(state_cost, np.concatenate([dt, traj.state_ddot[t]]))


def plan_swing_up(
    prog: MathematicalProgram,
    pc: tuple[MultibodyPlant, Context],
    pc_ad: tuple[MultibodyPlant, Context],
    prev_traj: Trajectory,
    index: int,
    num_timesteps=20,
) -> Trajectory:
    nq = pc[0].num_positions()
    nqd = pc[0].num_velocities()
    nu = pc[0].num_actuators()
    traj_out = Trajectory(
        t=prog.NewContinuousVariables(rows=num_timesteps, cols=1, name=f"s_dt_{index}"),
        state=prog.NewContinuousVariables(
            rows=num_timesteps + 1, cols=nq, name=f"s_q_{index}"
        ),
        state_dot=prog.NewContinuousVariables(
            rows=num_timesteps + 1, cols=nqd, name=f"s_q_dot_{index}"
        ),
        state_ddot=prog.NewContinuousVariables(
            rows=num_timesteps + 1, cols=nqd, name=f"s_q_ddot_{index}"
        ),
        control=prog.NewContinuousVariables(
            rows=num_timesteps + 1, cols=nu, name=f"s_u_{index}"
        ),
        contact_force=prog.NewContinuousVariables(
            rows=num_timesteps + 1, cols=2, name=f"s_f_{index}"
        ),
        is_successful=False,
    )
    # Add Constraints
    # Consistency with previous trajectory
    add_consistency_constraint(prog, prev_traj, traj_out)
    add_contact_constraints(
        pc, pc_ad, traj_out.state, traj_out.contact_force, num_timesteps, prog
    )
    add_joint_limit_constraints(
        pc, pc_ad, traj_out.state, traj_out.state_dot, traj_out.control, prog
    )
    add_time_step_constraints(traj_out.t, prog)
    add_trajectory_dynamics_constraints(pc, pc_ad, traj_out, prog)
    prog.AddBoundingBoxConstraint(
        [0.0, 0.0],
        [0.0, 0.0],
        traj_out.contact_force[-1],
    )

    add_trajectory_dynamics_constraints(pc, pc_ad, traj_out, prog)

    # Add Costs
    add_trajectory_costs(traj_out, prog)

    return traj_out


def add_hand_target_constraints(
    pc: tuple[MultibodyPlant, Context],
    pc_ad: tuple[MultibodyPlant, Context],
    traj: Trajectory,
    target_x_m: float,
    target_z_m: float,
    prog: MathematicalProgram,
):
    prog.AddConstraint(
        lambda q_t: frame_x_position(pc, pc_ad, q_t, "hand"),
        [target_x_m],
        [target_x_m],
        vars=traj.state[-1, :],
    )

    prog.AddConstraint(
        lambda q_t: frame_z_position(pc, pc_ad, q_t, "hand"),
        [target_z_m],
        [target_z_m],
        vars=traj.state[-1, :],
    )


def add_upright_collision_constraint(
    pc: tuple[MultibodyPlant, Context],
    pc_ad: tuple[MultibodyPlant, Context],
    traj: Trajectory,
    target_x_m: float,
    prog: MathematicalProgram,
):
    eps = 1e-3
    for t in range(traj.state.shape[0]):
        prog.AddConstraint(
            lambda q_t: frame_x_position(pc, pc_ad, q_t, "hand"),
            lb=[-np.inf],
            ub=[target_x_m + eps],
            vars=traj.state[t],
        )


def plan_flight(
    prog: MathematicalProgram,
    pc: tuple[MultibodyPlant, Context],
    pc_ad: tuple[MultibodyPlant, Context],
    prev_traj: Trajectory,
    target_x_m: float,
    target_z_m: float,
    index: int,
    num_timesteps=20,
) -> Trajectory:
    nq = pc[0].num_positions()
    nqd = pc[0].num_velocities()
    nu = pc[0].num_actuators()
    traj_out = Trajectory(
        t=prog.NewContinuousVariables(rows=num_timesteps, cols=1, name=f"f_dt_{index}"),
        state=prog.NewContinuousVariables(
            rows=num_timesteps + 1, cols=nq, name=f"f_q_{index}"
        ),
        state_dot=prog.NewContinuousVariables(
            rows=num_timesteps + 1, cols=nqd, name=f"f_q_dot_{index}"
        ),
        state_ddot=prog.NewContinuousVariables(
            rows=num_timesteps + 1, cols=nqd, name=f"f_q_ddot_{index}"
        ),
        control=prog.NewContinuousVariables(
            rows=num_timesteps + 1, cols=nu, name=f"f_u_{index}"
        ),
        contact_force=prog.NewContinuousVariables(
            rows=num_timesteps + 1, cols=2, name=f"f_f_{index}"
        ),
        is_successful=False,
    )
    # Add Constraints
    # Consistency with previous trajectory
    add_consistency_constraint(prog, prev_traj, traj_out)
    add_contact_constraints(pc, pc_ad, traj_out.state, traj_out.contact_force, 0, prog)
    add_joint_limit_constraints(
        pc, pc_ad, traj_out.state, traj_out.state_dot, traj_out.control, prog
    )
    add_positive_velocity_on_release_constraint(pc, pc_ad, traj_out.state_dot, 0, prog)
    add_time_step_constraints(traj_out.t, prog)
    add_trajectory_dynamics_constraints(pc, pc_ad, traj_out, prog)
    add_hand_target_constraints(pc, pc_ad, traj_out, target_x_m, target_z_m, prog)
    add_upright_collision_constraint(pc, pc_ad, traj_out, target_x_m, prog)

    # add costs
    add_trajectory_costs(traj_out, prog)

    return traj_out


def package_trajectory(result: MathematicalProgramResult, trajs: list[Trajectory]):
    fields = {"is_successful": result.is_success()}
    for field_name in Trajectory._fields:
        if field_name == "is_successful":
            continue
        fields[field_name] = []
        for traj in trajs:
            field = getattr(traj, field_name)
            if field is None:
                continue
            first_idx = 0 if (len(fields[field_name]) == 0 or field_name == "t") else 1
            fields[field_name].append(field[first_idx:])
        if len(fields[field_name]) > 1:
            fields[field_name] = result.GetSolution(np.concatenate(fields[field_name]))
        else:
            fields[field_name] = result.GetSolution(fields[field_name][0])
    return Trajectory(**fields)


def get_key_frames(
    pc: tuple[MultibodyPlant, Context],
    pc_ad: tuple[MultibodyPlant, Context],
    target_x_m: float,
    hand_height_targets_m: list[float],
):
    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(
        rows=len(hand_height_targets_m) * 2 - 1, cols=pc[0].num_positions()
    )

    for i, target_z_m in enumerate(hand_height_targets_m):
        # Add hanging constraints
        prog.AddConstraint(
            lambda q_t: [frame_x_position(pc, pc_ad, q_t, "hand")],
            [target_x_m],
            [target_x_m],
            vars=q[2 * i],
        )

        prog.AddConstraint(
            lambda q_t: [frame_z_position(pc, pc_ad, q_t, "hand")],
            [target_z_m],
            [target_z_m],
            vars=q[2 * i],
        )

        eps = 1e-3
        prog.AddConstraint(
            lambda q_t: [State(*q_t).torso_shoulder_joint_q],
            [np.pi - eps],
            [np.pi - eps],
            vars=q[2 * i],
        )

        prog.AddConstraint(
            lambda q_t: [State(*q_t).shoulder_hand_joint_x],
            [0.4],
            [0.4],
            vars=q[2 * i],
        )

        if i == len(hand_height_targets_m) - 1:
            break

        # Add launching constraints
        prog.AddConstraint(
            lambda q_t: [frame_x_position(pc, pc_ad, q_t, "hand")],
            [target_x_m],
            [target_x_m],
            vars=q[2 * i + 1],
        )

        prog.AddConstraint(
            lambda q_t: [frame_z_position(pc, pc_ad, q_t, "hand")],
            [target_z_m],
            [target_z_m],
            vars=q[2 * i + 1],
        )

        prog.AddConstraint(
            lambda q_t: [State(*q_t).torso_shoulder_joint_q],
            [np.pi / 2.0],
            [np.pi / 2.0],
            vars=q[2 * i + 1],
        )

        prog.AddConstraint(
            lambda q_t: [State(*q_t).shoulder_hand_joint_x],
            [0.2],
            [0.2],
            vars=q[2 * i + 1],
        )

    solver = SnoptSolver()
    result: MathematicalProgramResult = solver.Solve(prog)
    assert result.is_success()
    return result.GetSolution(q)


def get_initial_guess(
    pc: tuple[MultibodyPlant, Context],
    pc_ad: tuple[MultibodyPlant, Context],
    prog: MathematicalProgram,
    traj_vars: list[Trajectory],
    target_x_m: float,
    hand_height_targets_m: list[float],
) -> np.ndarray:
    out = np.zeros(prog.num_vars())

    key_frames = get_key_frames(pc, pc_ad, target_x_m, hand_height_targets_m)

    def lerp(a: State, b: State, frac: float):
        return State(*[(b[i] - a[i]) * frac + a[i] for i in range(len(a))])

    for i, traj in enumerate(traj_vars):
        if traj.t is not None:
            t_guess = np.ones_like(traj.t, dtype=np.float64) * 0.25
            prog.SetDecisionVariableValueInVector(traj.t, t_guess, out)
            nt = traj.state.shape[0]
            q_guess = np.zeros_like(traj.state)
            print('lerping between\n', key_frames[i-1], '\n', key_frames[i])
            for t in range(traj.state.shape[0]):
                q_guess[t, :] = lerp(key_frames[i-1], key_frames[i], t / nt)
            prog.SetDecisionVariableValueInVector(traj.state, q_guess, out)

    return out


def plan_ladder_climb(diagram: Diagram):
    prog = MathematicalProgram()

    root_context = diagram.CreateDefaultContext()
    plant = diagram.GetSubsystemByName("plant")
    context = plant.GetMyContextFromRoot(root_context)

    diagram_ad = diagram.ToAutoDiffXd()
    root_context_ad = diagram_ad.CreateDefaultContext()
    plant_ad = diagram_ad.GetSubsystemByName("plant")
    context_ad = plant_ad.GetMyContextFromRoot(root_context_ad)

    hand_height_targets_m = [2.11, 2.41, 2.71]
    target_x_m = -0.05

    trajectory_vars = [
        get_initial_state(
            prog,
            (plant, context),
            (plant_ad, context_ad),
            target_x_m,
            hand_height_targets_m[0],
        )
    ]

    for i, target_z_m in enumerate(hand_height_targets_m[1:]):
        trajectory_vars.append(
            plan_swing_up(
                prog, (plant, context), (plant_ad, context_ad), trajectory_vars[-1], i
            )
        )
        trajectory_vars.append(
            plan_flight(
                prog,
                (plant, context),
                (plant_ad, context_ad),
                trajectory_vars[-1],
                target_x_m,
                target_z_m,
                i,
            )
        )

    add_cycle_consistency_constraint(prog, trajectory_vars[-2], trajectory_vars[-1])
    initial_guess = get_initial_guess(
        (plant, context),
        (plant_ad, context_ad),
        prog,
        trajectory_vars,
        target_x_m,
        hand_height_targets_m,
    )

    solver = SnoptSolver()
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintFileName, "/tmp/solver.txt")
    options.SetOption(solver.id(), "Minor print level", "0")
    options.SetOption(solver.id(), "Iterations Limit", "200000")
    options.SetOption(solver.id(), "Solution", "No")
    result: MathematicalProgramResult = solver.Solve(
        prog, solver_options=options, initial_guess=initial_guess
    )
    print("Optimization Successful?", result.is_success())
    print(result.get_solution_result())
    out = package_trajectory(result, trajectory_vars)
    IPython.embed()
    return out
