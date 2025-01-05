import json
import robosuite as suite
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
# import pdb
import numpy as np

import os
import time, datetime
import gymnasium as gym
from OpenGL import GL

from utils import traverse_grid_3d, save_video
def search_for_place_cube_actions(rubik_x_size, rubik_y_size, rubik_z_size, target_cube_xyz_idx, red_cube_xyz_idx):
    actions = []
    # find the start pos
    start_cube_xyz_idx = None
    for z in range(rubik_z_size):
        for x in range(rubik_x_size - 1, -1, -1):
            for y in range(rubik_y_size):
                if target_cube_xyz_idx[x, y, z] == 1:
                    start_cube_xyz_idx = np.array([x, y, z])
                    break
            if start_cube_xyz_idx is not None:
                break
        if start_cube_xyz_idx is not None:
            break
    
    assert np.all(start_cube_xyz_idx == red_cube_xyz_idx), f"start_cube_xyz_idx: {start_cube_xyz_idx}, red_cube_xyz_idx: {red_cube_xyz_idx} should be the same"
    
    # find the path
    path = traverse_grid_3d(target_cube_xyz_idx, tuple(start_cube_xyz_idx))
    # print(f"path: {path}")
    
    placed_cube_xyz_idx = np.zeros_like(target_cube_xyz_idx)
    placed_cube_xyz_idx[start_cube_xyz_idx[0], start_cube_xyz_idx[1], start_cube_xyz_idx[2]] = 1
    # breakpoint()
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        delta_node = [next_node[0] - current_node[0], next_node[1] - current_node[1], next_node[2] - current_node[2]]
        delta_node_str = f"{delta_node[0]}_{delta_node[1]}_{delta_node[2]}"
        # direction
        if delta_node_str == "1_0_0":
            direction = "forward"
        elif delta_node_str == "-1_0_0":
            direction = "backward"
        elif delta_node_str == "0_1_0":
            direction = "right"
        elif delta_node_str == "0_-1_0":
            direction = "left"
        elif delta_node_str == "0_0_1":
            direction = "up"
        elif delta_node_str == "0_0_-1":
            direction = "down"
        else:
            # breakpoint()
            # raise NotImplementedError(f"Unknown delta_node_str: {delta_node_str}, Current node: {current_node}, Next node: {next_node}")
            pass
        
        # action
        if not placed_cube_xyz_idx[next_node[0], next_node[1], next_node[2]]:
            action = "place_block"
            placed_cube_xyz_idx[next_node[0], next_node[1], next_node[2]] = 1
        else:
            action = "move_cursor"
        actions.append(
            {
                "action": action,
                "direction": direction,
            }
        )
    
    return actions

from spatial_intelligence_wrapper import create_env
if __name__ == "__main__": 
    # test the sim speed of robosuite

    def ignore_gl_errors(*args, **kwargs):
        pass
    GL.glCheckError = ignore_gl_errors
    choose_robosuite = True
    save_video_flag = True
    
    env = create_env()
    eps = 1
    i_eps = 0
    step_count = 0
    start_time = time.time()
    images = []
    front_images = []
    agent_images = []
    robot_images = []
    
    env.reset()
    
    view_name = "frontview" # sideview, frontview, agentview, robot0_eye_in_hand, birdview
    while i_eps < eps:
        obs = env.reset()
        front_images.append(np.flipud(obs['obs'][view_name]))
        ep_step_count = 0
        for action in search_for_place_cube_actions(
            env.env.rubik_x_size, env.env.rubik_y_size, env.env.rubik_y_size, 
            env.cube_xyz_idx, env.env.rubik_red_cube_xyz_idx
        ):
            result, r, d, _ = env.step(action)
            obs = result["obs"]
            front_images.append(np.flipud(obs[view_name]))
            print(f"result: {result['success']}, message: {result['message']}")
            if "direction cannot be placed because there is no cube below it." in result['message']:
                raise NotImplementedError(f"action: {action}, result: {result['message']}")
            step_count += 1
        print(f"eps: {i_eps}, steps: {step_count}, blocks success: {r}")
        i_eps += 1
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_step = total_time / step_count
    # print(f"time_per_step: {time_per_step}")
    
    # save video 
    if save_video_flag:
        print(f"save video...")
        os.makedirs("./video", exist_ok=True)
        images.extend(front_images)
        save_video(f"./video/SpatialIntelligence_{view_name}.mp4", [images])
