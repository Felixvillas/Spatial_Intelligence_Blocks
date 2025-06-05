import numpy as np
from PIL import Image
import sys, os
import matplotlib.pyplot as plt
from copy import deepcopy
import math
from noise import pnoise3
import pickle as pkl
import random

from robosuite.environments.manipulation.spatial_intelligence import SpatialIntelligence
import time
import tqdm

class SpatialIntelligenceWrapper:
    def __init__(
        self,
        env,
        task="connected_cube",
        extra_params={},
    ):
        assert isinstance(env, SpatialIntelligence)
        self.env = env
        self.views = deepcopy(env.camera_names)
        # self.operate_view = "sideview"
        # self.target_view = self.operate_view
        # self.target_view = "frontview"
        self.reset_vars()
        self.episode_max_len = 200
        self.available_tasks = [
            "mrt", # connected blocks in Mental Rotation Test,
        ]
        self.task = task
        self.extra_params = extra_params
        if self.task not in self.available_tasks:
            raise NotImplementedError(f"Unknown task: {self.task}")
        if self.task == "mrt":
            mrt_data_path = "mrt.pkl"
            if os.path.exists(mrt_data_path):
                self.mrt_data = pkl.load(open("mrt.pkl", "rb"))
            else:
                self.mrt_data = None

        self.mrtviews = [
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
            

    def reset_vars(self):
        # self.perspective_view = "frontview"
        self.target_blocks_views = []
        self.solutions = []
        self.solution = None
        self.cube_xyz_idxs = []
        self.cube_xyz_idx = None
        self.paths = []
        self.path = None
        self.episode_step = 0
        
    def reset(self):
        self.reset_vars()
        self.generate_task()
        self.env.reset()
        self.move_red_cube_to_init_pos()
        # step to get the observation
        obs, r, d, _ = self.env.step(np.zeros([]))
        return {
            "obs": {
                view: np.flipud(obs[f"{view}_image"]) for view in self.mrtviews
            },
            "target_block": random.choice(self.target_blocks_views),
        }
    
    def step(self, action):
        act, direction = action["action"], action["direction"]
        if act == "switch_view":
            raise NotImplementedError("switch_view is not implemented")
        elif act == "move_cursor":
            results = self.env.move_red_cube(direction)
        elif act == "place_block":
            results = self.env.place_cube(direction, self.cube_xyz_idx)
        else:
            raise NotImplementedError(f"Unknown action: {act}")
        
        obs, r, d, _ = self.env.step(np.zeros([]))
        self.episode_step += 1
        results["obs"] = {
            view: np.flipud(obs[f"{view}_image"]) for view in self.mrtviews
        }
        success = self.check_success()
        if success:
            r = 1
            d = True
        else:
            r = 0
            d = False
        if self.episode_step >= self.episode_max_len:
            d = True
        results["target_block"] = random.choice(self.target_blocks_views)
        return results, r, d, {}
    
    def check_success(self):
        # for cube_xyz_idx in self.cube_xyz_idxs:
        #     if np.all(self.env.rubik_xyz_idx_exists == cube_xyz_idx):
        #         return True
        
        if np.all(self.env.rubik_xyz_idx_exists == self.cube_xyz_idx):
            return True
        return False
    
    def move_red_cube_to_init_pos(self):
        red_cube_xyz_idx = np.array(self.path[0])
        # set the red cube to the init position
        self.env.set_cube_joint(
            cube_name="virtual_cube_red_cube",
            cube_pos=self.env.final_rubik_position[f"{red_cube_xyz_idx[0]}_{red_cube_xyz_idx[1]}_{red_cube_xyz_idx[2]}"],
        )
        # set the red cube's xyz_idx and rubik_xyz_idx_exists in the env
        self.env.rubik_red_cube_xyz_idx = red_cube_xyz_idx
        self.env.rubik_xyz_idx_exists[red_cube_xyz_idx[0], red_cube_xyz_idx[1], red_cube_xyz_idx[2]] = True
        return red_cube_xyz_idx
    
    
    def consider_gravity(self, cube_xyz_idx):
        for z in range(cube_xyz_idx.shape[2]):
            for y in range(cube_xyz_idx.shape[1]):
                for x in range(cube_xyz_idx.shape[0]):
                    if cube_xyz_idx[x, y, z] == 1:
                        for down_level in range(z - 1, -1, -1):
                            if cube_xyz_idx[x, y, down_level] == 0:
                                cube_xyz_idx[x, y, down_level] = 1
                                cube_xyz_idx[x, y, down_level + 1] = 0
        
        return cube_xyz_idx
    def assert_gravity(self, cube_xyz_idx):
        for z in range(cube_xyz_idx.shape[2]):
            for y in range(cube_xyz_idx.shape[1]):
                for x in range(cube_xyz_idx.shape[0]):
                    if cube_xyz_idx[x, y, z] == 1:
                        for down_level in range(z - 1, -1, -1):
                            assert cube_xyz_idx[x, y, down_level] == 1, f"cube_xyz_idx[x, y, down_level]: {cube_xyz_idx[x, y, down_level]} should be 1"
                            
    def generate_rubik_by_cube_xyz_idx(self, cube_xyz_idx):
            # generate rubik by referring to the cube
            if self.env.is_gravity:
                cube_xyz_idx = self.consider_gravity(cube_xyz_idx)
                self.assert_gravity(cube_xyz_idx)
            
            self.env.reset()
            """
                Generate a rubik's cube by the xyz index of the red cube.
            """
            rubik_x_size, rubik_y_size, rubik_z_size = cube_xyz_idx.shape
            assert (rubik_x_size == self.env.rubik_x_size and rubik_y_size == self.env.rubik_y_size and rubik_z_size == self.env.rubik_z_size)
            
            for z in range(rubik_z_size):
                for y in range(rubik_y_size):
                    for x in range(rubik_x_size):
                        if cube_xyz_idx[x, y, z]:
                            self.env.set_cube_joint(
                                cube_name=f"virtual_cube_{x}_{y}_{z}",
                                cube_pos=self.env.final_rubik_position[f"{x}_{y}_{z}"],
                            )
            # for target cube's init: move the red cube to the virtual_cube_0_0_0's init position
            self.env.set_cube_joint(
                cube_name="virtual_cube_red_cube",
                cube_pos=self.env.initial_rubik_position["0_0_0"],
            )
            self.env.sim.forward()
            obs, r, d, _ = self.env.step(np.zeros([]))
            return obs
        
    def generate_task(self):
        
        if self.task == "mrt":
            assert not self.env.is_gravity, "is_gravity should be False when task is mrt"
            assert self.mrt_data is not None, "self.mrt_data should not be None"
            self.generate_mrt()
        else:
            raise NotImplementedError(f"Unknown task: {self.task}")
        
    def generate_mrt(self, number_of_blocks=None):
        if number_of_blocks is None:
            number_of_blocks = random.choice(list(self.mrt_data.keys()))
        # selected_comb = random.choice(list(self.mrt_data[number_of_blocks].keys()))
        # origin_or_mirror = random.choice(["origin", "mirror"])
        # all_mrts = self.mrt_data[number_of_blocks][selected_comb][origin_or_mirror]
        # self.target_blocks_views = [all_mrts[k]["mrtview_image"] for k in all_mrts.keys()]
        # self.solutions = [all_mrts[k]["solution"] for k in all_mrts.keys()]
        # self.cube_xyz_idxs = [all_mrts[k]["cube_xyz_idx"] for k in all_mrts.keys()]
        # self.paths = [all_mrts[k]["path"] for k in all_mrts.keys()]
        
        # specific_mrt_idx = random.choice(range(len(self.solutions)))
        # self.solution = self.solutions[specific_mrt_idx]
        # self.cube_xyz_idx = self.cube_xyz_idxs[specific_mrt_idx]
        # self.path = self.paths[specific_mrt_idx]

        specific_mrt_idx = random.choice(range(len(self.mrt_data[number_of_blocks])))
        origin_or_mirror = random.choice([0, 1]) # 0 is origin, 1 is mirror
        self.solution = self.mrt_data[number_of_blocks][specific_mrt_idx][origin_or_mirror]["solution"]
        self.cube_xyz_idx = self.mrt_data[number_of_blocks][specific_mrt_idx][origin_or_mirror]["cube_xyz_idx"]
        self.path = self.mrt_data[number_of_blocks][specific_mrt_idx][origin_or_mirror]["path"]
        self.target_blocks_views = [self.mrt_data[number_of_blocks][specific_mrt_idx][origin_or_mirror][view] for view in self.mrtviews]
        
    
    def reset_collect_mrt_sft_data(self):
        self.episode_step = 0
        self.env.reset()
        self.move_red_cube_to_init_pos()
        # step to get the observation
        obs, r, d, _ = self.env.step(np.zeros([]))
        return {
            "obs": {
                view: np.flipud(obs[f"{view}_image"]) for view in self.mrtviews
            },
            "target_block": random.choice(self.target_blocks_views),
        }
        
    # def collect_mrt_sft_data(self):
    #     for number_of_block in self.mrt_data.keys():
    #         for comb in self.mrt_data[number_of_block].keys():
    #             for origin_or_mirror in ["origin", "mirror"]:
    #                 mrtview_images = [
    #                     self.mrt_data[number_of_block][comb][origin_or_mirror][i]["mrtview_image"] \
    #                         for i in self.mrt_data[number_of_block][comb][origin_or_mirror].keys()
    #                 ]
    #                 xyz_orders = [
    #                     self.mrt_data[number_of_block][comb][origin_or_mirror][i]["xyz_order"] \
    #                         for i in self.mrt_data[number_of_block][comb][origin_or_mirror].keys()
    #                 ]
    #                 for idx in self.mrt_data[number_of_block][comb][origin_or_mirror].keys():
    #                     xyz_order = self.mrt_data[number_of_block][comb][origin_or_mirror][idx]["xyz_order"]
    #                     self.path = self.mrt_data[number_of_block][comb][origin_or_mirror][idx]["path"]
    #                     self.solution = self.mrt_data[number_of_block][comb][origin_or_mirror][idx]["solution"]
    #                     self.cube_xyz_idx = self.mrt_data[number_of_block][comb][origin_or_mirror][idx]["cube_xyz_idx"]
    #                     mrtview_image = self.mrt_data[number_of_block][comb][origin_or_mirror][idx]["mrtview_image"]
    #                     self.target_blocks_views = [mrtview_image] # actual useless, just for dont raise an exception in reset and step function
    #                     yield {
    #                         "number_of_block": number_of_block,
    #                         "comb": comb,
    #                         "origin_or_mirror": origin_or_mirror,
    #                         "idx": idx,
    #                         "xyz_order": xyz_order,
    #                         "path": self.path,
    #                         "solution": self.solution,
    #                         "cube_xyz_idx": self.cube_xyz_idx,
    #                         "mrtview_image": mrtview_image,
    #                         "mrtview_images": mrtview_images,
    #                         "xyz_orders": xyz_orders
    #                     }

    def collect_mrt_sft_data(self):
        for number_of_block in self.mrt_data.keys():
            for item in self.mrt_data[number_of_block]:
                origin = item[0]
                mirror = item[1]

                for idx, om in enumerate([origin, mirror]):
                    self.path = om["path"]
                    self.solution = om["solution"]
                    self.cube_xyz_idx = om["cube_xyz_idx"]
                    return_dict = {
                        "number_of_block": number_of_block,
                        "comb": om["comb"],
                        "origin_or_mirror": "origin" if idx == 0 else "mirror",
                        # "xyz_order": om["xyz_order"],
                        "path": self.path,
                        "solution": self.solution,
                        "cube_xyz_idx": self.cube_xyz_idx,
                    }
                    return_dict.update(
                        {view : om[view] for view in self.mrtviews}
                    )
                    self.target_blocks_views = [om[view] for view in self.mrtviews]
                    yield return_dict
        
        
        
    # def generate_all_mrt(self, number_of_blocks):
    #     def cal_all_combs(total):
    #         a = []
    #         for i in range(2, total+1):
    #             for j in range(2, total+1):
    #                 for k in range(2, total+1):
    #                     for l in range(2, total+1):
    #                         if i+j+k+l == total:
    #                             a.append((i, j, k, l))
            
    #         return set(a)
        
    #     def is_same_traj(traj1, traj2):
    #         def traverse(traj):
    #             path = []
    #             for idx in range(len(traj)-1):
    #                 vec = traj[idx + 1] - traj[idx]
    #                 assert np.bool_(vec).sum() == 1, f"should only 1 element is different from 0 in vec, vec: {vec}"
    #                 path.append(abs(vec[np.nonzero(vec)[0][0]]))
                    
    #             vec1 = traj[1] - traj[0] # actual the x axis
    #             vec2 = traj[2] - traj[1] # actual the y axis
    #             z = np.cross(vec1, vec2) / np.linalg.norm(np.cross(vec1, vec2)) # actual the z axis
    #             vec3 = traj[3] - traj[2] 
    #             d = np.dot(vec3, z)
    #             if d > 0:
    #                 path.extend(["+x", "+y", "+z", "+x"])
    #             elif d < 0:
    #                 path.extend(["+x", "+y", "-z", "+x"])
    #             else:
    #                 raise ValueError(f"d should be either greater than 0 or less than 0, but it is {d}")
                
    #             return path
            
    #         path1 = traverse(traj1)
    #         path2 = traverse(traj2)
    #         path2_reversed = traverse(list(reversed(traj2)))
    #         return path1 == path2 or path1 == path2_reversed
        
    #     def tarjs_from_ls(l, choice = None, max_xyz=7):
    #         # item in l should be greater than or equal to 2
    #         assert all([item >= 2 for item in l]), f"all items in l should be greater than or equal to 2, l: {l}"
    #         # boundary check
    #         assert l[0] + l[3] <= max_xyz, f"l[0] + l[3] should be less than or equal to {max_xyz}, but it is {l[0] + l[3]}"
    #         assert l[1] <= max_xyz, f"l[1] should be less than or equal to {max_xyz}, but it is {l[1]}"
    #         assert l[2] <= max_xyz, f"l[2] should be less than or equal to {max_xyz}, but it is {l[2]}"
            
    #         add_sub_dict = {"+": 1, "-": -1}
    #         xyz_idx_dict = {"x": 0, "y": 1, "z": 2}
    #         start_pos = [-1, -1, -1]
    #         if choice[0][0] == "+":
    #             start_pos[xyz_idx_dict[choice[0][1]]] = 0
    #         elif choice[0][0] == "-":
    #             start_pos[xyz_idx_dict[choice[0][1]]] = l[0] - 1 + l[3] - 1
    #         else:
    #             raise ValueError(f"choice[0][0] should be either '+' or '-'")
            
    #         if choice[1][0] == "+":
    #             start_pos[xyz_idx_dict[choice[1][1]]] = 0
    #         elif choice[1][0] == "-":
    #             start_pos[xyz_idx_dict[choice[1][1]]] = l[1] - 1
    #         else:
    #             raise ValueError(f"choice[1][0] should be either '+' or '-'")
            
    #         if choice[2][0] == "+":
    #             start_pos[xyz_idx_dict[choice[2][1]]] = 0
    #         elif choice[2][0] == "-":
    #             start_pos[xyz_idx_dict[choice[2][1]]] = l[2] - 1
    #         else:
    #             raise ValueError(f"choice[2][0] should be either '+' or '-'")
            
    #         assert start_pos[0] != -1 and start_pos[1] != -1 and start_pos[2] != -1, f"start_pos: {start_pos}"
            
    #         traj = [start_pos]
    #         for idx, path in enumerate(choice):
    #             pos = deepcopy(traj[-1])
    #             pos[xyz_idx_dict[path[1]]] += add_sub_dict[path[0]] * (l[idx] - 1)
    #             traj.append(deepcopy(pos))
    #         return traj
        
    #     def all_trajs_from_ls(l, max_xyz):
    
    #         # item in l should be greater than or equal to 2
    #         try:
    #             assert all([item >= 2 for item in l]), f"all items in l should be greater than or equal to 2, l: {l}"
    #             # boundary check
    #             assert l[0] + l[3] <= max_xyz, f"l[0] + l[3] should be less than or equal to {max_xyz}, but it is {l[0] + l[3]}"
    #             assert l[1] <= max_xyz, f"l[1] should be less than or equal to {max_xyz}, but it is {l[1]}"
    #             assert l[2] <= max_xyz, f"l[2] should be less than or equal to {max_xyz}, but it is {l[2]}"
    #         except Exception as e:
    #             return [], []
            
    #         choice_candidates = [
    #             ["+x", "+y", "+z", "+x"],
    #             ["+x", "+y", "-z", "+x"],
    #             ["+x", "-y", "+z", "+x"],
    #             ["+x", "-y", "-z", "+x"],
    #             ["+x", "+z", "+y", "+x"],
    #             ["+x", "+z", "-y", "+x"],
    #             ["+x", "-z", "+y", "+x"],
    #             ["+x", "-z", "-y", "+x"],
    #             ["-x", "+y", "+z", "-x"],
    #             ["-x", "+y", "-z", "-x"],
    #             ["-x", "-y", "+z", "-x"],
    #             ["-x", "-y", "-z", "-x"],
    #             ["-x", "+z", "+y", "-x"],
    #             ["-x", "+z", "-y", "-x"],
    #             ["-x", "-z", "+y", "-x"],
    #             ["-x", "-z", "-y", "-x"],
    #             ["+y", "+x", "+z", "+y"],
    #             ["+y", "+x", "-z", "+y"],
    #             ["+y", "-x", "+z", "+y"],
    #             ["+y", "-x", "-z", "+y"],
    #             ["+y", "+z", "+x", "+y"],
    #             ["+y", "+z", "-x", "+y"],
    #             ["+y", "-z", "+x", "+y"],
    #             ["+y", "-z", "-x", "+y"],
    #             ["-y", "+x", "+z", "-y"],
    #             ["-y", "+x", "-z", "-y"],
    #             ["-y", "-x", "+z", "-y"],
    #             ["-y", "-x", "-z", "-y"],
    #             ["-y", "+z", "+x", "-y"],
    #             ["-y", "+z", "-x", "-y"],
    #             ["-y", "-z", "+x", "-y"],
    #             ["-y", "-z", "-x", "-y"],
    #             ["+z", "+x", "+y", "+z"],
    #             ["+z", "+x", "-y", "+z"],
    #             ["+z", "-x", "+y", "+z"],
    #             ["+z", "-x", "-y", "+z"],
    #             ["+z", "+y", "+x", "+z"],
    #             ["+z", "+y", "-x", "+z"],
    #             ["+z", "-y", "+x", "+z"],
    #             ["+z", "-y", "-x", "+z"],
    #             ["-z", "+x", "+y", "-z"],
    #             ["-z", "+x", "-y", "-z"],
    #             ["-z", "-x", "+y", "-z"],
    #             ["-z", "-x", "-y", "-z"],
    #             ["-z", "+y", "+x", "-z"],
    #             ["-z", "+y", "-x", "-z"],
    #             ["-z", "-y", "+x", "-z"],
    #             ["-z", "-y", "-x", "-z"],
    #         ]
    #         trajs_1 = []
    #         trajs_2 = []
    #         traj_reference = tarjs_from_ls(l, choice=choice_candidates[0])
            
    #         for choice in choice_candidates:
    #             traj = tarjs_from_ls(l, choice=choice)
    #             if is_same_traj(np.array(traj_reference), np.array(traj)):
    #                 trajs_1.append({"xyz_order": "_".join(choice), "traj": deepcopy(traj)})
    #             else:
    #                 trajs_2.append({"xyz_order": "_".join(choice), "traj": deepcopy(traj)})
    #         return trajs_1, trajs_2
        
    #     def path_from_pos1_to_pos2(pos1, pos2):
    #         vec = pos2 - pos1
    #         assert np.bool_(vec).sum() == 1, f"should only 1 element is different from 0 in vec, vec: {vec}"
    #         nonzero_idx = np.nonzero(vec)[0][0]
    #         path = []
    #         if vec[nonzero_idx] > 0:
    #             for i in range(pos1[nonzero_idx], pos2[nonzero_idx] + 1):
    #                 path.append(deepcopy(pos1))
    #                 path[-1][nonzero_idx] = i
    #         elif vec[nonzero_idx] < 0:
    #             for i in range(pos1[nonzero_idx], pos2[nonzero_idx] - 1, -1):
    #                 path.append(deepcopy(pos1))
    #                 path[-1][nonzero_idx] = i
    #         else:
    #             raise ValueError(f"vec[nonzero_idx] should be either greater than 0 or less than 0, but it is {vec[nonzero_idx]}")
    #         return path
        
    #     def search_from_path_for_mrt(path):
    #         def delta_str2direction(delta_str):
    #             if delta_str == "1_0_0":
    #                 direction = "forward"
    #             elif delta_str == "-1_0_0":
    #                 direction = "backward"
    #             elif delta_str == "0_1_0":
    #                 direction = "right"
    #             elif delta_str == "0_-1_0":
    #                 direction = "left"
    #             elif delta_str == "0_0_1":
    #                 direction = "up"
    #             elif delta_str == "0_0_-1":
    #                 direction = "down"
    #             elif delta_str == "0_0_0":
    #                 direction = None
    #             else:
    #                 raise NotImplementedError(f"Unknown delta_str: {delta_str}")
    #             return direction
            
    #         actions = []
    #         for idx in range(len(path) - 1):
    #             del_pos = [path[idx + 1][0] - path[idx][0], path[idx + 1][1] - path[idx][1], path[idx + 1][2] - path[idx][2]]
    #             del_pos_str = f"{del_pos[0]}_{del_pos[1]}_{del_pos[2]}"
    #             direction = delta_str2direction(del_pos_str)
    #             if direction is None:
    #                 continue
    #             actions.append(
    #                 {
    #                     "action": "place_block",
    #                     "direction": direction,
    #                 }
    #             )
                
    #         return actions
                
        
    #     data = {}
    #     data[number_of_blocks] = {}
    #     all_combs = list(cal_all_combs(number_of_blocks))
    #     # breakpoint()
    #     # random.shuffle(adds)
    #     for comb in tqdm.tqdm(all_combs):
    #         add_str = "_".join([str(item) for item in comb])
    #         trajs_1, trajs_2 = all_trajs_from_ls(comb, max_xyz=7)
            
    #         if trajs_1 == [] or trajs_2 == []:
    #             continue
            
    #         data[number_of_blocks][add_str] = {}
            
    #         data[number_of_blocks][add_str]["origin"] = {}
    #         data[number_of_blocks][add_str]["mirror"] = {}
    #         n = self.env.rubik_x_size
    #         for trajs_idx, trajs in enumerate([trajs_1, trajs_2]):
    #             for traj_idx, traj_dict in enumerate(trajs):
    #                 xyz_order = traj_dict["xyz_order"]
    #                 traj = traj_dict["traj"]
    #                 path = []
    #                 for idx in range(len(traj) - 1):
    #                     path.extend(path_from_pos1_to_pos2(np.array(traj[idx]), np.array(traj[idx + 1])))
                    
    #                 actual_path = []
    #                 for pos_idx in range(len(path)):
    #                     if pos_idx == len(path) - 1:
    #                         actual_path.append(path[pos_idx])
    #                     else:
    #                         if not np.array_equal(path[pos_idx], path[pos_idx + 1]):
    #                             actual_path.append(path[pos_idx])
                    
    #                 solution = search_from_path_for_mrt(actual_path)
                    
    #                 cube_xyz_idx = np.zeros((n, n, n), dtype=int)
    #                 for item in actual_path:
    #                     cube_xyz_idx[item[0], item[1], item[2]] = 1
                        
    #                 mrtview_image = np.flipud(self.generate_rubik_by_cube_xyz_idx(cube_xyz_idx)["mrtview_image"])
                    
    #                 if trajs_idx == 0:
    #                     data[number_of_blocks][add_str]["origin"][traj_idx] = {
    #                         "xyz_order": xyz_order,
    #                         "path": actual_path,
    #                         "solution": solution,
    #                         "cube_xyz_idx": cube_xyz_idx,
    #                         "mrtview_image": mrtview_image
    #                     }
    #                 elif trajs_idx == 1:
    #                     data[number_of_blocks][add_str]["mirror"][traj_idx] = {
    #                         "xyz_order": xyz_order,
    #                         "path": actual_path,
    #                         "solution": solution,
    #                         "cube_xyz_idx": cube_xyz_idx,
    #                         "mrtview_image": mrtview_image
    #                     }
    #                 else:
    #                     raise NotImplementedError
                    
    #     assert data != {}, f"data is empty dict"
    #     # save data as a pickle
    #     pkl.dump(obj=data, file=open(f"mrt_{number_of_blocks}.pkl", "wb"))

    def generate_all_mrt(self, number_of_blocks):
        def cal_all_combs(total):
            a = []
            for i in range(2, total+1):
                for j in range(2, total+1):
                    for k in range(2, total+1):
                        for l in range(2, total+1):
                            if i+j+k+l == total:
                                a.append((i, j, k, l))
            
            return set(a)
        
        def is_same_traj(traj1, traj2):
            def traverse(traj):
                path = []
                for idx in range(len(traj)-1):
                    vec = traj[idx + 1] - traj[idx]
                    assert np.bool_(vec).sum() == 1, f"should only 1 element is different from 0 in vec, vec: {vec}"
                    path.append(abs(vec[np.nonzero(vec)[0][0]]))
                    
                vec1 = traj[1] - traj[0] # actual the x axis
                vec2 = traj[2] - traj[1] # actual the y axis
                z = np.cross(vec1, vec2) / np.linalg.norm(np.cross(vec1, vec2)) # actual the z axis
                vec3 = traj[3] - traj[2] 
                d = np.dot(vec3, z)
                if d > 0:
                    path.extend(["+x", "+y", "+z", "+x"])
                elif d < 0:
                    path.extend(["+x", "+y", "-z", "+x"])
                else:
                    raise ValueError(f"d should be either greater than 0 or less than 0, but it is {d}")
                
                return path
            
            path1 = traverse(traj1)
            path2 = traverse(traj2)
            path2_reversed = traverse(list(reversed(traj2)))
            return path1 == path2 or path1 == path2_reversed
        
        def tarjs_from_ls(l, choice = None, max_xyz=7):
            # item in l should be greater than or equal to 2
            assert all([item >= 2 for item in l]), f"all items in l should be greater than or equal to 2, l: {l}"
            # boundary check
            assert l[0] + l[3] <= max_xyz + 1, f"l[0] + l[3] should be less than or equal to {max_xyz + 1}, but it is {l[0] + l[3]}"
            assert l[1] <= max_xyz, f"l[1] should be less than or equal to {max_xyz}, but it is {l[1]}"
            assert l[2] <= max_xyz, f"l[2] should be less than or equal to {max_xyz}, but it is {l[2]}"
            
            add_sub_dict = {"+": 1, "-": -1}
            xyz_idx_dict = {"x": 0, "y": 1, "z": 2}
            start_pos = [-1, -1, -1]
            if choice[0][0] == "+":
                start_pos[xyz_idx_dict[choice[0][1]]] = 0
            elif choice[0][0] == "-":
                start_pos[xyz_idx_dict[choice[0][1]]] = l[0] - 1 + l[3] - 1
            else:
                raise ValueError(f"choice[0][0] should be either '+' or '-'")
            
            if choice[1][0] == "+":
                start_pos[xyz_idx_dict[choice[1][1]]] = 0
            elif choice[1][0] == "-":
                start_pos[xyz_idx_dict[choice[1][1]]] = l[1] - 1
            else:
                raise ValueError(f"choice[1][0] should be either '+' or '-'")
            
            if choice[2][0] == "+":
                start_pos[xyz_idx_dict[choice[2][1]]] = 0
            elif choice[2][0] == "-":
                start_pos[xyz_idx_dict[choice[2][1]]] = l[2] - 1
            else:
                raise ValueError(f"choice[2][0] should be either '+' or '-'")
            
            assert start_pos[0] != -1 and start_pos[1] != -1 and start_pos[2] != -1, f"start_pos: {start_pos}"
            
            traj = [start_pos]
            for idx, path in enumerate(choice):
                pos = deepcopy(traj[-1])
                pos[xyz_idx_dict[path[1]]] += add_sub_dict[path[0]] * (l[idx] - 1)
                traj.append(deepcopy(pos))
            return traj
        
        def all_trajs_from_ls(l, max_xyz):
    
            # item in l should be greater than or equal to 2
            try:
                assert all([item >= 2 for item in l]), f"all items in l should be greater than or equal to 2, l: {l}"
                # boundary check
                assert l[0] + l[3] <= max_xyz + 1, f"l[0] + l[3] should be less than or equal to {max_xyz + 1}, but it is {l[0] + l[3]}"
                assert l[1] <= max_xyz, f"l[1] should be less than or equal to {max_xyz}, but it is {l[1]}"
                assert l[2] <= max_xyz, f"l[2] should be less than or equal to {max_xyz}, but it is {l[2]}"
            except Exception as e:
                return {}, {}
            
            choice_candidates = [
                ["+x", "+y", "+z", "+x"], # origin
                ["+x", "+y", "-z", "+x"], # mirror
            ]
            origin = tarjs_from_ls(l, choice=choice_candidates[0])
            mirror = tarjs_from_ls(l, choice=choice_candidates[1])
            origin = {
                "number_of_blocks": number_of_blocks - 3,
                "comb": "_".join([str(item) if idx == 0 else str(item - 1) for idx, item in enumerate(l)]),
                "structure": "origin",
                "traj": origin
            }
            mirror = {
                "number_of_blocks": number_of_blocks - 3,
                "comb": "_".join([str(item) if idx == 0 else str(item - 1) for idx, item in enumerate(l)]),
                "structure": "mirror",
                "traj": mirror
            }
            return origin, mirror
        
        def path_from_pos1_to_pos2(pos1, pos2):
            vec = pos2 - pos1
            assert np.bool_(vec).sum() == 1, f"should only 1 element is different from 0 in vec, vec: {vec}"
            nonzero_idx = np.nonzero(vec)[0][0]
            path = []
            if vec[nonzero_idx] > 0:
                for i in range(pos1[nonzero_idx], pos2[nonzero_idx] + 1):
                    path.append(deepcopy(pos1))
                    path[-1][nonzero_idx] = i
            elif vec[nonzero_idx] < 0:
                for i in range(pos1[nonzero_idx], pos2[nonzero_idx] - 1, -1):
                    path.append(deepcopy(pos1))
                    path[-1][nonzero_idx] = i
            else:
                raise ValueError(f"vec[nonzero_idx] should be either greater than 0 or less than 0, but it is {vec[nonzero_idx]}")
            return path
        
        def search_from_path_for_mrt(path):
            def delta_str2direction(delta_str):
                if delta_str == "1_0_0":
                    direction = "forward"
                elif delta_str == "-1_0_0":
                    direction = "backward"
                elif delta_str == "0_1_0":
                    direction = "right"
                elif delta_str == "0_-1_0":
                    direction = "left"
                elif delta_str == "0_0_1":
                    direction = "up"
                elif delta_str == "0_0_-1":
                    direction = "down"
                elif delta_str == "0_0_0":
                    direction = None
                else:
                    raise NotImplementedError(f"Unknown delta_str: {delta_str}")
                return direction
            
            actions = []
            for idx in range(len(path) - 1):
                del_pos = [path[idx + 1][0] - path[idx][0], path[idx + 1][1] - path[idx][1], path[idx + 1][2] - path[idx][2]]
                del_pos_str = f"{del_pos[0]}_{del_pos[1]}_{del_pos[2]}"
                direction = delta_str2direction(del_pos_str)
                if direction is None:
                    continue
                actions.append(
                    {
                        "action": "place_block",
                        "direction": direction,
                    }
                )
                
            return actions
                
        
        data = []
        all_combs = list(cal_all_combs(number_of_blocks))
        # breakpoint()
        # random.shuffle(adds)
        for comb in tqdm.tqdm(all_combs):
            temp_data = []
            origin, mirror = all_trajs_from_ls(comb, max_xyz=7)
            
            if origin == {} or mirror == {}:
                continue

            n = self.env.rubik_x_size

            for traj_idx, traj_dict in enumerate([origin, mirror]):
                traj = traj_dict["traj"]
                path = []
                for idx in range(len(traj) - 1):
                    path.extend(path_from_pos1_to_pos2(np.array(traj[idx]), np.array(traj[idx + 1])))
                
                actual_path = []
                for pos_idx in range(len(path)):
                    if pos_idx == len(path) - 1:
                        actual_path.append(path[pos_idx])
                    else:
                        if not np.array_equal(path[pos_idx], path[pos_idx + 1]):
                            actual_path.append(path[pos_idx])
                
                solution = search_from_path_for_mrt(actual_path)
                
                cube_xyz_idx = np.zeros((n, n, n), dtype=int)
                for item in actual_path:
                    cube_xyz_idx[item[0], item[1], item[2]] = 1
                    
                obs = self.generate_rubik_by_cube_xyz_idx(cube_xyz_idx)
                
                if traj_idx == 0:
                    origin_copy = deepcopy(origin)
                    origin_copy.update(
                        {
                            "path": actual_path,
                            "solution": solution,
                            "cube_xyz_idx": cube_xyz_idx,
                        }
                    )
                    origin_copy.update(
                        {
                            view : np.flipud(obs[f"{view}_image"]) for view in self.mrtviews
                        }
                    )
                    temp_data.append(origin_copy)
                elif traj_idx == 1:
                    mirror_copy = deepcopy(mirror)
                    mirror_copy.update(
                        {
                            "path": actual_path,
                            "solution": solution,
                            "cube_xyz_idx": cube_xyz_idx,
                        }
                    )
                    mirror_copy.update(
                        {
                            view : np.flipud(obs[f"{view}_image"]) for view in self.mrtviews
                        }
                    )
                    temp_data.append(mirror_copy)
                else:
                    raise NotImplementedError
            data.append(temp_data)
        assert data != [], f"data is empty dict"
        # save data as a pickle
        pkl.dump(obj=data, file=open(f"mrt_{number_of_blocks - 3}.pkl", "wb"))
        
        
                
                

        
import robosuite as suite
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
def create_env(task="connected_cube", extra_params={}):
    if task == "generated_connected_cube":
        assert extra_params != {}, "extra_params should not be \{\} when task is generated_connected_cube"
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
        "render_camera": [],
        "camera_names": [
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

            # "frontview", 
            # "topview", 
            # "sideview"
        ], 
        "camera_depths": True,
        "camera_heights": extra_params["width"], # 512
        "camera_widths": extra_params["width"], # 512
        "reward_shaping": True,
        "has_renderer": False,
        "use_object_obs": True,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "is_gravity": False,
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
        task=task,
        # task="connected_cube", # see available_tasks in SpatialIntelligenceWrapper
        # task="spherical_surface",
        # task="perlin_noise",
        extra_params=extra_params,
    )
    return env