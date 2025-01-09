import json
# import pdb
import numpy as np

import os
import time, datetime
import gymnasium as gym
from OpenGL import GL

from utils import traverse_grid_3d, save_video, search_for_place_cube_actions

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
