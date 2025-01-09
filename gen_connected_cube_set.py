from utils import traverse_grid_3d, save_video, search_for_place_cube_actions
from spatial_intelligence_wrapper import create_env
from copy import deepcopy
import pickle as pkl


env = create_env(task="connected_cube")
num_connected_cube = 1000
eps = 1

connected_cube_set = {}

while eps <= num_connected_cube:
    obs = env.reset()
    actions = search_for_place_cube_actions(
        env.env.rubik_x_size, env.env.rubik_y_size, env.env.rubik_y_size, 
        env.cube_xyz_idx, env.env.rubik_red_cube_xyz_idx
    )
    ground_actions = []
    for action in actions:
        result, r, d, _ = env.step(action)
        ground_actions.append(action)
        obs = result["obs"]
        # print(f"result: {result['success']}, message: {result['message']}")
        if "direction cannot be placed because there is no cube below it." in result['message']:
            raise NotImplementedError(f"action: {action}, result: {result['message']}")
        if r or d:
            break
    if r:
        print(f"eps: {eps}, blocks success: {r}")
        connected_cube_set[f"task_{eps}"] = {
            "task": deepcopy(env.cube_xyz_idx),
            "solution": deepcopy(ground_actions)
        }
        eps += 1
    else:
        print(f"eps: {eps}, blocks success: {r}, retry...")
        
        
# save connect cubes set to pkl file
pkl.dump(connected_cube_set, open(f"connected_cube_set_{num_connected_cube}.pkl", "wb"))
