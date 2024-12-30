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
            action = "place"
            placed_cube_xyz_idx[next_node[0], next_node[1], next_node[2]] = 1
        else:
            action = "move"
        actions.append(
            {
                "action": action,
                "direction": direction,
            }
        )
    
    return actions

from spatial_intelligence_wrapper import SpatialIntelligenceWrapper
if __name__ == "__main__": 
    # test the sim speed of robosuite

    def ignore_gl_errors(*args, **kwargs):
        pass
    GL.glCheckError = ignore_gl_errors
    choose_robosuite = True
    save_video_flag = True
    
    env_config = {
        "control_freq": 20,
        "controller": "OSC_POSE",
        "env_name": "SpatialIntelligence",
        "hard_reset": True,
        "horizon": 500,
        "ignore_done": True,
        "reward_scale": 1.0,
        "robots": "UR5e",
        "gripper_types": "default",
        "render_gpu_device_id": 0,
        "render_camera": ["frontview"],
        "camera_names": [
            "top2bottom", "bottom2top",
            "sideview_0", "sideview_45", "sideview_90", "sideview_135",
            "sideview_180", "sideview_225", "sideview_270", "sideview_315",
        ], 
        "camera_depths": True,
        "camera_heights": 1024,
        "camera_widths": 1024,
        "reward_shaping": True,
        "has_renderer": False,
        "use_object_obs": True,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "is_gravity": True,
    }
    # Load controller
    controller = env_config.pop("controller")
    if controller in set(ALL_CONTROLLERS):
        # This is a default controller
        controller_config = load_controller_config(default_controller=controller)
    else:
        # This is a string to the custom controller
        controller_config = load_controller_config(custom_fpath=controller)
    controller_config['control_delta'] = True
    controller_config['uncouple_pos_ori'] = True
    
    env_suite = suite.make(
        **env_config,
        controller_configs=controller_config,
    )
    
    env = env_suite
    
    env = SpatialIntelligenceWrapper(
        env, 
        task="connected_cube", # see available_tasks in SpatialIntelligenceWrapper
        # task="spherical_surface",
        # task="perlin_noise",
    )
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
        front_images.append(obs["perspective_view"])
        ep_step_count = 0
        for action in search_for_place_cube_actions(
            env.env.rubik_x_size, env.env.rubik_y_size, env.env.rubik_y_size, 
            env.cube_xyz_idx, env.env.rubik_red_cube_xyz_idx
        ):
            result, r, d, _ = env.step(action)
            obs = result["obs"]
            front_images.append(obs["perspective_view"])
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
        save_video(f"./video/{env_config['env_name']}_{env.perspective_view}.mp4", [images])
