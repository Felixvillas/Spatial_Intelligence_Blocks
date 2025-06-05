import json
import numpy as np

import os
import time, datetime
import gymnasium as gym
# from OpenGL import GL

import cv2

from utils import traverse_grid_3d, save_video, search_for_think_and_answer_v1

from spatial_intelligence_wrapper import create_env

def save_image(image_array, image_path):
    # cv2 is BGR not RGB
    cv2.imwrite(image_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    # #or using plt
    # plt.imsave(image_path, image_array)
    # return the abs image path
    return os.path.abspath(image_path)

if __name__ == "__main__": 

    # def ignore_gl_errors(*args, **kwargs):
    #     pass
    # GL.glCheckError = ignore_gl_errors
    choose_robosuite = True
    save_video_flag = True
    
    env = create_env(task="mrt", extra_params={"width": 256})
    # env = create_env(task="connected_cube", extra_params={"width": 256, "is_gravity": True})
    # 创建一个 7×7×7 的全零数组
    arr = np.zeros((7, 7, 7), dtype=int)

    # 随机选择 10 个位置设置为 1
    # total_elements = 7 * 7 * 7 = 343
    indices = np.random.choice(7*7*7, 10, replace=False)

    # 将这些位置设置为 1
    arr[np.unravel_index(indices, arr.shape)] = 1
    
    cube_xyz_idx = arr
    
    obs = env.generate_rubik_by_cube_xyz_idx(cube_xyz_idx)
    
    views = [
            "frontview", "topview", "sideview",
            "mrtview_0",
            "mrtview_45",
            "mrtview_90",
            "mrtview_135",
            "mrtview_180",
            "mrtview_225",
            "mrtview_270",
            "mrtview_315",

            "mrtview_30",
            "mrtview_60",
            "mrtview_120",
            "mrtview_150",
            "mrtview_210",
            "mrtview_240",
            "mrtview_300",
            "mrtview_330",
        ]
    for view in views:
        # breakpoint()
        save_image(np.flipud(obs[f"{view}_image"]), f"./image_views/{view}.png")
    

