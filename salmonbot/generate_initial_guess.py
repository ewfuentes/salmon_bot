from pydrake.all import (
    MultibodyPlant,
    Context,
    MathematicalProgram,
    MathematicalProgramResult,
    SnoptSolver,
)

from salmonbot.trajectory_planner import (
    State,
    Trajectory,
    frame_x_position,
    frame_z_position,
)

import numpy as np


def get_key_frames(
    pc: tuple[MultibodyPlant, Context],
    pc_ad: tuple[MultibodyPlant, Context],
    target_x_m: float,
    hook_length_x_m: float,
    hook_length_z_m: float,
    free_above_hook_clearance_m: float,
    hand_height_targets_m: list[float],
):
    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(
        rows=len(hand_height_targets_m) * 4 - 3, cols=pc[0].num_positions()
    )

    eps = 0.01
    for i, target_z_m in enumerate(hand_height_targets_m):
        # Add hanging constraints
        prog.AddConstraint(
            lambda q_t: [frame_x_position(pc, pc_ad, q_t, "hand")],
            [target_x_m],
            [target_x_m],
            vars=q[4 * i],
        )

        prog.AddConstraint(
            lambda q_t: [frame_z_position(pc, pc_ad, q_t, "hand")],
            [target_z_m],
            [target_z_m],
            vars=q[4 * i],
        )

        eps = 1e-3
        prog.AddConstraint(
            lambda q_t: [State(*q_t).torso_shoulder_joint_q],
            [np.pi - eps],
            [np.pi - eps],
            vars=q[4 * i],
        )

        prog.AddConstraint(
            lambda q_t: [State(*q_t).shoulder_hand_joint_x],
            [0.4],
            [0.4],
            vars=q[4 * i],
        )

        if i == len(hand_height_targets_m) - 1:
            break

        # Add take off constraints
        prog.AddConstraint(
            lambda q_t: [frame_x_position(pc, pc_ad, q_t, "hand")],
            [target_x_m],
            [target_x_m],
            vars=q[4 * i + 1],
        )

        prog.AddConstraint(
            lambda q_t: [frame_z_position(pc, pc_ad, q_t, "hand")],
            [target_z_m],
            [target_z_m],
            vars=q[4 * i + 1],
        )

        prog.AddConstraint(
            lambda q_t: [State(*q_t).torso_shoulder_joint_q],
            [np.pi / 2.0],
            [np.pi / 2.0],
            vars=q[4 * i + 1],
        )

        prog.AddConstraint(
            lambda q_t: [State(*q_t).shoulder_hand_joint_x],
            [0.2],
            [0.2],
            vars=q[4 * i + 1],
        )

        # Add clear hook x constraints
        prog.AddConstraint(
            lambda q_t: [frame_x_position(pc, pc_ad, q_t, "hand")],
            [target_x_m + hook_length_x_m - eps],
            [target_x_m + hook_length_x_m - eps],
            vars=q[4 * i + 2],
        )

        prog.AddConstraint(
            lambda q_t: [frame_z_position(pc, pc_ad, q_t, "hand")],
            [target_z_m + free_above_hook_clearance_m],
            [target_z_m + free_above_hook_clearance_m],
            vars=q[4 * i + 2],
        )

        prog.AddConstraint(
            lambda q_t: [State(*q_t).torso_shoulder_joint_q],
            [np.pi],
            [np.pi],
            vars=q[4 * i + 2],
        )

        prog.AddConstraint(
            lambda q_t: [State(*q_t).shoulder_hand_joint_x],
            [0.4],
            [0.4],
            vars=q[4 * i + 2],
        )

        # Add clear hook z constraints
        prog.AddConstraint(
            lambda q_t: [frame_x_position(pc, pc_ad, q_t, "hand")],
            [target_x_m + hook_length_x_m - eps],
            [target_x_m + hook_length_x_m - eps],
            vars=q[4 * i + 3],
        )

        prog.AddConstraint(
            lambda q_t: [frame_z_position(pc, pc_ad, q_t, "hand")],
            [target_z_m + 0.3 + hook_length_z_m],
            [target_z_m + 0.3 + hook_length_z_m],
            vars=q[4 * i + 3],
        )

        prog.AddConstraint(
            lambda q_t: [State(*q_t).torso_shoulder_joint_q],
            [np.pi],
            [np.pi],
            vars=q[4 * i + 3],
        )

        prog.AddConstraint(
            lambda q_t: [State(*q_t).shoulder_hand_joint_x],
            [0.4],
            [0.4],
            vars=q[4 * i + 3],
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
    hook_length_x_m: float,
    hook_length_z_m: float,
    free_above_hook_clearance_m: float,
    hand_height_targets_m: list[float],
) -> np.ndarray:
    out = np.zeros(prog.num_vars())

    key_frames = get_key_frames(
        pc,
        pc_ad,
        target_x_m,
        hook_length_x_m,
        hook_length_z_m,
        free_above_hook_clearance_m,
        hand_height_targets_m,
    )

    def lerp(a: State, b: State, frac: float):
        return State(*[(b[i] - a[i]) * frac + a[i] for i in range(len(a))])

    for i, traj in enumerate(traj_vars):
        if traj.t is not None:
            t_guess = np.ones_like(traj.t, dtype=np.float64) * 0.25
            prog.SetDecisionVariableValueInVector(traj.t, t_guess, out)
            nt = traj.state.shape[0]
            q_guess = np.zeros_like(traj.state)
            for t in range(traj.state.shape[0]):
                q_guess[t, :] = lerp(key_frames[i - 1], key_frames[i], t / nt)
            prog.SetDecisionVariableValueInVector(traj.state, q_guess, out)

    return out
