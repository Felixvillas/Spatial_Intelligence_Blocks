import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cv2
import numpy as np
import json
import datetime, time
from Spatial_Intelligence_Blocks.utils import save_video, search_for_place_cube_actions



def save_image(image_array, image_path):
    # cv2 is BGR not RGB
    cv2.imwrite(image_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    # #or using plt
    # plt.imsave(image_path, image_array)
    # return the abs image path
    return os.path.abspath(image_path)

    
def save_data_as_video(datas, save_path):
    os.makedirs(save_path, exist_ok=True)
    images = []
    for datas_eps in datas:
        for obs, _, _, _ in datas_eps:
            images.append(np.flipud(obs['obs']['frontview']))
    save_video(os.path.join(save_path, "video.mp4"), [images])
    print(f"Video saved to {save_path}")
    
def save_target_blocks(env, save_path):
    
    os.makedirs(os.path.join(save_path, "target_blocks"), exist_ok=True)
    target_blocks_dirs = [item for item in os.listdir(os.path.join(save_path, "target_blocks")) if os.path.isdir(os.path.join(save_path, "target_blocks", item)) and item.startswith("target_blocks")]
    target_blocks_idxs = sorted([int(item.split("_")[-1]) for item in target_blocks_dirs])
    if target_blocks_idxs == []:
        target_blocks_idx = 0
    else:
        target_blocks_idx = target_blocks_idxs[-1] + 1
    for k, v in env.target_blocks_views.items():
        os.makedirs(os.path.join(save_path, "target_blocks", f"target_blocks_{target_blocks_idx}"), exist_ok=True)
        # plt.imsave(os.path.join(save_path, "target_blocks", f"target_blocks_{target_blocks_idx}", f"{k}.png"), v)
        cv2.imwrite(os.path.join(save_path, "target_blocks", f"target_blocks_{target_blocks_idx}", f"{k}.png"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
        

def main(steps):
    start_time = time.time()
    start_time_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    # task = "connected_cube"
    task = "generated_connected_cube"
    save_path = os.path.join(f"./temp/{task}/{start_time_str}")
    from Spatial_Intelligence_Blocks.spatial_intelligence_wrapper import create_env
    env = create_env(task)
    i_steps = 0
    datas = []
    while i_steps < steps:
        obs = env.reset()
        save_target_blocks(env, save_path)
        done = False
        datas_eps = []
        if task == "generated_connected_cube":
            actions = env.generated_connected_cube_set_solution
        else:
            actions = search_for_place_cube_actions(
                env.env.rubik_x_size, env.env.rubik_y_size, env.env.rubik_y_size, 
                env.cube_xyz_idx, env.env.rubik_red_cube_xyz_idx
            )
        for action in actions:
            next_obs, reward, done, info = env.step(action)
            # append the tuple (obs, messages, response, next_obs, done) to datas
            response = "```json\n{\n"
            response += f"\"Action\": {json.dumps(action, indent=4)}\n"
            response += "}\n```"
            datas_eps.append((obs, response, next_obs, done))
            
            i_steps += 1
            obs = next_obs
            print(f"Step {i_steps}: Act: {action}, Success: {next_obs['success']}, Message: {next_obs['message']}")
            if done:
                print(f"blocks success: {reward}")
                break
            
        datas.append(datas_eps)
        if task == "generated_connected_cube":
            if env.generated_connected_cube_set_idx > 100: # change it to 1000 if you test 1000set
                break
    
    end_time = time.time()
    print(f"time per step: {(end_time - start_time) / steps}")
    
    save_data_as_video(datas, save_path)
    # save the datas
    # save_data(datas, save_path)
    return save_path
    
        
if __name__ == "__main__":
    data_path = main(
        steps=20000
    )