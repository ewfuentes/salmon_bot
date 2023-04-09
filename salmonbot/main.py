#!/usr/bin/env python3
import argparse

from pydrake.all import MultibodyPlant, MeshcatVisualizer, Parser, ModelVisualizer, Simulator, StartMeshcat


def run(world_path: str, robot_path: str):
    print('Loading model from', world_path)
    print('Loading model from', robot_path)

    meshcat = StartMeshcat()
    visualizer = ModelVisualizer(meshcat=meshcat)
    visualizer.AddModels(world_path)
    visualizer.AddModels(robot_path)
    visualizer.Run(loop_once=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--robot', help='path to robot URDF model', required=True)
    parser.add_argument(
        '--world', help='path to world URDF model', required=True)

    args = parser.parse_args()

    run(args.world, args.robot)
