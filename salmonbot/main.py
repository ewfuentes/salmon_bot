#!/usr/bin/env python3
import argparse

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Context,
    StartMeshcat,
    Meshcat,
    Diagram,
    DiagramBuilder,
    Parser,
    MultibodyPlant,
    SceneGraph,
    RigidTransform,
    MeshcatVisualizer,
    Simulator,
    RollPitchYaw,
    InitializeParams,
)

import numpy as np
import time
import IPython

from salmonbot.trajectory_planner import plan_trajectory, Trajectory

np.set_printoptions(linewidth=200)


def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)


def load_model(world_path: str, robot_path: str, meshcat: MeshcatVisualizer):
    builder = DiagramBuilder()
    result: tuple[MultibodyPlant, SceneGraph] = AddMultibodyPlantSceneGraph(
        builder, time_step=0.0
    )
    plant, scene_graph = result
    parser = Parser(plant)
    parser.AddModelFromFile(robot_path, model_name="Robot")
    parser.AddModelFromFile(world_path, model_name="World")

    plant.WeldFrames(
        frame_on_parent_F=plant.GetFrameByName("hand"),
        frame_on_child_M=plant.GetFrameByName("bar"),
        X_FM=xyz_rpy_deg(np.array([0.0, 0.0, 0.0]), np.array([90.0, 0.0, 0.0])),
    )

    plant.WeldFrames(
        frame_on_parent_F=plant.world_frame(),
        frame_on_child_M=plant.GetFrameByName("ground"),
        X_FM=RigidTransform.Identity(),
    )

    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    plant.Finalize()
    diagram = builder.Build()
    return diagram, scene_graph


def set_initial_conditions(diagram: Diagram, context: Context):
    plant: MultibodyPlant = diagram.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    plant.SetPositions(
        plant_context,
        np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,  # world_from_torso quaternion
                -0.05,
                0.0,
                1.8,  # world_from_torso translations
                3.14,  # Torso - shoulder joint
                0.38,  # shoulder - hand joint
                0.0,  # torso upper_leg joint
                0.0,  # upper_leg lower_leg joint
            ]
        ),
    )

    plant.get_actuation_input_port().FixValue(
        plant_context,
        np.array(
            [
                -100.0,  # arm_force
                0.0,  # shoulder_torque
                0.0,  # hip torque
                0.0,  # knee torque
            ]
        ),
    )


def visualize_trajectory(
    meshcat: Meshcat,
    trajectory: Trajectory,
    simulator: Simulator,
    diagram: Diagram,
    context: Context,
):
    t = [0] + list(np.cumsum(trajectory.t))
    STOP_TRAJECTORY_VIZ_STR = "Stop Trajectory Viz"
    TIMESTEP_SLIDER_STR = "Timestep Selector"
    meshcat.AddButton(STOP_TRAJECTORY_VIZ_STR)
    meshcat.AddSlider(TIMESTEP_SLIDER_STR, 0, len(t) - 1, step=1.0, value=0.0)
    plant: MultibodyPlant = diagram.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)
    simulator.Initialize()

    run = True
    prev_step_value = -1.0
    while run:
        # get inputs
        num_times_stopped_pressed = meshcat.GetButtonClicks(STOP_TRAJECTORY_VIZ_STR)
        step_value = meshcat.GetSliderValue(TIMESTEP_SLIDER_STR)

        # set state
        run = num_times_stopped_pressed == 0
        should_redraw = False

        if prev_step_value != step_value:
            prev_step_value = step_value
            should_redraw = True
            idx = int(step_value)
            plant.SetPositions(plant_context, trajectory.state[idx])
            plant.SetVelocities(plant_context, trajectory.state_dot[idx])
            # context.SetTime(t[idx])
            print(f"{t[idx]}: {trajectory.state[idx]}")

        # flush
        if should_redraw:
            simulator.Initialize(InitializeParams(suppress_initialization_events=True))
        time.sleep(0.1)
    meshcat.DeleteButton(STOP_TRAJECTORY_VIZ_STR)


def run(world_path: str, robot_path: str):
    meshcat = StartMeshcat()
    # Build a diagram
    diagram, scene_graph = load_model(world_path, robot_path, meshcat)

    # Create the simulator
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    # Set the initial conditions
    set_initial_conditions(diagram, context)

    # Plan a trajectory
    trajectory = plan_trajectory(diagram)

    print(
        f"Visualizing Resulting Trajectory. Is successful? {trajectory.is_successful}"
    )
    visualize_trajectory(meshcat, trajectory, simulator, diagram, context)

    # Build a controller

    # Run the simulation
    # simulator.Initialize()
    # simulator.set_publish_every_time_step(True)
    # simulator.set_target_realtime_rate(0.25)
    # input("press enter to continue")
    # for i in np.arange(0, 5.0, 0.25):
    #     print(f"Sim step: {i}")
    #     simulator.AdvanceTo(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", help="path to robot URDF model", required=True)
    parser.add_argument("--world", help="path to world URDF model", required=True)

    args = parser.parse_args()

    run(args.world, args.robot)
