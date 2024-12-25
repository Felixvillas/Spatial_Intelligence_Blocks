import json
import robosuite as suite
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
# import pdb
import numpy as np

import os
import imageio
import time, datetime
import gymnasium as gym
from OpenGL import GL

def save_video(video_path_name, datas, fps=20):
    if not video_path_name.endswith(".mp4"):
        video_path_name += ".mp4"
    
    video_writer = imageio.get_writer(video_path_name, fps=fps)
    for data in datas:
        for img in data:
            video_writer.append_data(img[::-1])
            
    video_writer.close()
    
def place_cube_actions(rubik_x_size, rubik_y_size, rubik_z_size):
    actions = []
    for z in range(rubik_z_size):
        for y in range(rubik_y_size):
            for x in range(rubik_x_size - 1):
                actions.append(
                    {
                        "action": "place",
                        "direction": "forward",
                    }
                )
            for x in range(rubik_x_size - 1):
                actions.append(
                    {
                        "action": "move",
                        "direction": "backward",
                    }
                )
            if y < rubik_y_size - 1:
                actions.append(
                    {
                        "action": "place",
                        "direction": "right",
                    }
                )
            else:
                for y in range(rubik_y_size - 1):
                    actions.append(
                        {
                            "action": "move",
                            "direction": "left",
                        }
                    )
                    
        if z < rubik_z_size - 1:
            actions.append(
                {
                    "action": "place",
                    "direction": "up",
                }
            )
            
    # print(actions)
            
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
        # breakpoint()
        front_images.append(obs["perspective_view"])
        print(f"eps: {i_eps}, steps: {step_count}")
        ep_step_count = 0
        for action in place_cube_actions(env.env.rubik_x_size, env.env.rubik_y_size, env.env.rubik_y_size):
            result, r, d, _ = env.step(action)
            obs = result["obs"]
            front_images.append(obs["perspective_view"])
            print(f"result: {result['success']}, message: {result['message']}")
            step_count += 1
        i_eps += 1
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_step = total_time / step_count
    print(f"time_per_step: {time_per_step}")
    
    # save video 
    if save_video_flag:
        print(f"save video")
        os.makedirs("./video", exist_ok=True)
        images.extend(front_images)
        save_video(f"./video/{env_config['env_name']}_{env.perspective_view}.mp4", [images])
