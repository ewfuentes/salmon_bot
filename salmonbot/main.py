#!/usr/bin/env python3
import argparse

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ModelVisualizer,
    StartMeshcat,
    DiagramBuilder,
    Parser,
)


def load_model(world_path: str, robot_path: str):
    builder = DiagramBuilder()
    robot, robot_scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(robot).AddModelFromFile(robot_path, model_name="Robot")
    Parser(robot).AddModelFromFile(world_path, model_name="World")


def run(world_path: str, robot_path: str):
    load_model(world_path, robot_path)
    meshcat = StartMeshcat()
    visualizer = ModelVisualizer(meshcat=meshcat)
    visualizer.AddModels(world_path)
    visualizer.AddModels(robot_path)
    visualizer.Run(loop_once=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", help="path to robot URDF model", required=True)
    parser.add_argument("--world", help="path to world URDF model", required=True)

    args = parser.parse_args()

    run(args.world, args.robot)
