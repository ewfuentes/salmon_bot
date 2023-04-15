#!/usr/bin/env python3
import argparse

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    StartMeshcat,
    DiagramBuilder,
    Parser,
    MultibodyPlant,
    SceneGraph,
    RigidTransform,
    MeshcatVisualizer,
    Simulator,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    CameraInfo,
    RenderCameraCore,
    ClippingRange,
    ColorRenderCamera,
    DepthRenderCamera,
    RgbdSensor,
    RollPitchYaw,
    DepthRange,
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
        X_FM=RigidTransform.Identity(),
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


def run(world_path: str, robot_path: str):
    meshcat = StartMeshcat()
    # Build a diagram
    diagram, scene_graph = load_model(world_path, robot_path, meshcat)

    # Set the initial conditions

    # Plan a trajectory

    # Build a controller

    # Run the simulation

    Simulator(diagram).Initialize()

    time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", help="path to robot URDF model", required=True)
    parser.add_argument("--world", help="path to world URDF model", required=True)

    args = parser.parse_args()

    run(args.world, args.robot)
