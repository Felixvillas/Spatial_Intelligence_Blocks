from utils import traverse_grid_3d, save_video, search_for_place_cube_actions, search_for_think_and_answer, search_for_think_and_answer_v1
from spatial_intelligence_wrapper import create_env
from copy import deepcopy
import pickle as pkl
import numpy as np


env = create_env(task="connected_cube", extra_params={})
num_connected_cube = 1250
eps = 1
eps_success = 0
eps_fail = 0

connected_cube_set = {}

while eps <= num_connected_cube:
    obs = env.reset()
    if any(np.array_equal(env.cube_xyz_idx, item["task"]) for item in connected_cube_set.values()):
        print(f"task already exists, skip...")
        continue
    actions, thinks = search_for_think_and_answer_v1(
        env.env.rubik_x_size, env.env.rubik_y_size, env.env.rubik_y_size, 
        env.cube_xyz_idx, env.env.rubik_red_cube_xyz_idx
    )
    ground_actions = []
    for action in actions:
        result, r, d, _ = env.step(action)
        ground_actions.append(action)
        obs = result["obs"]
        print(f"result: {result['success']}, message: {result['message']}")
        if "direction cannot be placed because there is no cube below it." in result['message']:
            raise NotImplementedError(f"action: {action}, result: {result['message']}")
        if r or d:
            break
    if r:
        print(f"eps: {eps}, blocks success: {r}")
        connected_cube_set[f"task_{eps}"] = {
            "task": deepcopy(env.cube_xyz_idx),
            "solution": deepcopy(ground_actions),
            "think": deepcopy(thinks)
        }
        eps += 1
        eps_success += 1
        # breakpoint()
    else:
        eps_fail += 1
        print(f"eps: {eps}, blocks success: {r}, retry...")
        
print(f"eps_success: {eps_success}, eps_fail: {eps_fail}, success_rate: {eps_success / (eps_success + eps_fail)}")
        
# split connected_cube_set into train and test
train_connected_cube_set = {}
test_connected_cube_set = {}
for idx, (key, value) in enumerate(connected_cube_set.items()):
    if idx < int(num_connected_cube * 0.8):
        train_connected_cube_set[key] = value
    else:
        test_connected_cube_set[key] = value

connected_cube_set = {
    "train": train_connected_cube_set,
    "test": test_connected_cube_set
}
# save connect cubes set to pkl file
pkl.dump(connected_cube_set, open(f"connected_cube_set_{num_connected_cube}_traintest-w_thinks.pkl", "wb"))
