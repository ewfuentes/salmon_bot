#!/usr/bin/env python3
import argparse

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    StartMeshcat,
    Diagram,
    DiagramBuilder,
    Parser,
    MultibodyPlant,
    SceneGraph,
    RigidTransform,
    MeshcatVisualizer,
    System,
    Simulator,
    RollPitchYaw,
)
import numpy as np
import time


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


def set_initial_conditions(simulator: Simulator, diagram: Diagram):
    context = simulator.get_mutable_context()
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



def run(world_path: str, robot_path: str):
    meshcat = StartMeshcat()
    # Build a diagram
    diagram, scene_graph = load_model(world_path, robot_path, meshcat)

    # Create the simulator
    simulator = Simulator(diagram)

    # Set the initial conditions
    set_initial_conditions(simulator, diagram)

    # Plan a trajectory

    # Build a controller

    # Run the simulation
    simulator.Initialize()
    simulator.set_publish_every_time_step(True)
    simulator.set_target_realtime_rate(0.25)
    input('press enter to continue')
    for i in np.arange(0, 5.0, 0.25):
        print(f"Sim step: {i}")
        simulator.AdvanceTo(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", help="path to robot URDF model", required=True)
    parser.add_argument("--world", help="path to world URDF model", required=True)

    args = parser.parse_args()

    run(args.world, args.robot)
