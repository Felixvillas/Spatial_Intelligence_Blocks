import numpy as np
from PIL import Image
import sys, os
import matplotlib.pyplot as plt
from copy import deepcopy
import math
from noise import pnoise3

from robosuite.environments.manipulation.spatial_intelligence import SpatialIntelligence

class SpatialIntelligenceWrapper:
    def __init__(
        self,
        env,
        task="connected_cube",
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
            "connected_cube", # connected blocks generated by DFS
            "spherical_surface", # blocks on the spherical surface,
            "perlin_noise", # blocks generated by perlin noise
        ]
        self.task = task

        
    def reset_vars(self):
        # self.perspective_view = "frontview"
        self.target_blocks_views = {
            key: None for key in self.views
        }
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
                # "perspective_view": np.flipud(obs[f"{self.perspective_view}_image"]),
                # "operate_view": np.flipud(obs[f"{self.operate_view}_image"]),
                "frontview": np.flipud(obs["frontview_image"]),
                "topview": np.flipud(obs["topview_image"]),
                "sideview": np.flipud(obs["sideview_image"]),
            },
            # "target_block": self.target_blocks_views[self.target_view],
            "target_block": {
                "frontview": self.target_blocks_views["frontview"],
                "topview": self.target_blocks_views["topview"],
                "sideview": self.target_blocks_views["sideview"],
            }
        }
    
    def step(self, action):
        act, direction = action["action"], action["direction"]
        if act == "switch_view":
            raise NotImplementedError("switch_view is not implemented")
            if direction not in self.views:
                results = {
                    "obs": {
                        # "perspective_view": np.flipud(obs[f"{self.perspective_view}_image"]),
                        # "operate_view": np.flipud(obs[f"{self.operate_view}_image"]),
                        "frontview": np.flipud(obs["frontview_image"]),
                        "topview": np.flipud(obs["topview_image"]),
                        "sideview": np.flipud(obs["sideview_image"]),
                    },
                    # "target_block": self.target_blocks_views[self.target_view],
                    "target_block": {
                        "frontview": self.target_blocks_views["frontview"],
                        "topview": self.target_blocks_views["topview"],
                        "sideview": self.target_blocks_views["sideview"],
                    },
                    "success": False,
                    "message": f"Unknown view: {direction}",
                }
                return results, 0, False, {}
            self.perspective_view = direction
            results = {
                "success": True,
                "message": f"Switched to view: {direction}",
            }
        elif act == "move_cursor":
            results = self.env.move_red_cube(direction)
        elif act == "place_block":
            results = self.env.place_cube(direction, self.cube_xyz_idx)
        else:
            raise NotImplementedError(f"Unknown action: {act}")
        
        obs, r, d, _ = self.env.step(np.zeros([]))
        self.episode_step += 1
        results["obs"] = {
            # "perspective_view": np.flipud(obs[f"{self.perspective_view}_image"]),
            # "operate_view": np.flipud(obs[f"{self.operate_view}_image"]),
            "frontview": np.flipud(obs["frontview_image"]),
            "topview": np.flipud(obs["topview_image"]),
            "sideview": np.flipud(obs["sideview_image"]),
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
        # results["target_block"] = self.target_blocks_views[self.target_view]
        results["target_block"] = {
            "frontview": self.target_blocks_views["frontview"],
            "topview": self.target_blocks_views["topview"],
            "sideview": self.target_blocks_views["sideview"],
        }
        return results, r, d, {}
    
    def check_success(self):
        if np.all(self.env.rubik_xyz_idx_exists == self.cube_xyz_idx):
            return True
        return False
    
    def move_red_cube_to_init_pos(self):
        # find the init position of the red cube by self.cube_xyz_idx: first min to max, from z to x to y
        red_cube_xyz_idx = None
        for z in range(self.cube_xyz_idx.shape[2]):
            for x in range(self.cube_xyz_idx.shape[0] - 1, -1, -1):
                for y in range(self.cube_xyz_idx.shape[1]):
                    if self.cube_xyz_idx[x, y, z] == 1:
                        red_cube_xyz_idx = np.array([x, y, z])
                        break
                if red_cube_xyz_idx is not None:
                    break
            if red_cube_xyz_idx is not None:
                break
        # set the red cube to the init position
        self.env.set_cube_joint(
            cube_name="virtual_cube_red_cube",
            cube_pos=self.env.final_rubik_position[f"{red_cube_xyz_idx[0]}_{red_cube_xyz_idx[1]}_{red_cube_xyz_idx[2]}"],
        )
        # set the red cube's xyz_idx and rubik_xyz_idx_exists in the env
        self.env.rubik_red_cube_xyz_idx = red_cube_xyz_idx
        self.env.rubik_xyz_idx_exists[red_cube_xyz_idx[0], red_cube_xyz_idx[1], red_cube_xyz_idx[2]] = True
        return red_cube_xyz_idx
    
    def generate_task(self):
        def generate_rubik_by_cube_xyz_idx(cube_xyz_idx):
            # generate rubik by referring to the cube
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
                            
            self.move_red_cube_to_init_pos()
                            
            self.env.sim.forward()
            obs, r, d, _ = self.env.step(np.zeros([]))
            return obs
        
        if self.task == "connected_cube":
            cube_xyz_idx = self.generate_connected_cube(
                np.random.randint(
                    low=self.env.rubik_x_size + 1, 
                    high=self.env.rubik_x_size * 2
                )
            )
        elif self.task == "spherical_surface":
            cube_xyz_idx = self.generate_spherical_surface(
                np.random.randint(
                    low=1, 
                    high=self.env.rubik_x_size // 2 + 2
                )
            )
        elif self.task == "perlin_noise":
            cube_xyz_idx = self.generate_perlin_noise(
                np.random.randint(
                    low=3, 
                    high=self.env.rubik_x_size + 1
                )
            )
            print(f"cube_xyz_idx: {cube_xyz_idx.sum()}")
        else:
            raise NotImplementedError(f"Unknown task: {self.task}")
        
        def consider_gravity(cube_xyz_idx):
            for z in range(cube_xyz_idx.shape[2]):
                for y in range(cube_xyz_idx.shape[1]):
                    for x in range(cube_xyz_idx.shape[0]):
                        if cube_xyz_idx[x, y, z] == 1:
                            for down_level in range(z - 1, -1, -1):
                                if cube_xyz_idx[x, y, down_level] == 0:
                                    cube_xyz_idx[x, y, down_level] = 1
                                    cube_xyz_idx[x, y, down_level + 1] = 0
            
            return cube_xyz_idx
        def assert_gravity(cube_xyz_idx):
            for z in range(cube_xyz_idx.shape[2]):
                for y in range(cube_xyz_idx.shape[1]):
                    for x in range(cube_xyz_idx.shape[0]):
                        if cube_xyz_idx[x, y, z] == 1:
                            for down_level in range(z - 1, -1, -1):
                                assert cube_xyz_idx[x, y, down_level] == 1, f"cube_xyz_idx[x, y, down_level]: {cube_xyz_idx[x, y, down_level]} should be 1"
        # breakpoint()
        # consider the gravity
        if self.env.is_gravity:
            cube_xyz_idx = consider_gravity(cube_xyz_idx)
            assert_gravity(cube_xyz_idx)
        
        # breakpoint()                   
        self.cube_xyz_idx = cube_xyz_idx
        obs = generate_rubik_by_cube_xyz_idx(cube_xyz_idx)
        for view in self.views:
            self.target_blocks_views[view] = np.flipud(obs[f"{view}_image"])
            os.makedirs("task_view", exist_ok=True)
            plt.imsave(f"task_view/{view}.png", self.target_blocks_views[view])

    def generate_connected_cube(self, number_of_blocks):
        # 初始化魔方矩阵，0代表空，1代表积木块
        n = self.env.rubik_x_size
        cube_xyz_idx = np.zeros((n, n, n), dtype=int)
        # choice [0, 0, 0] as the start point
        start_x, start_y, start_z = [0, 0, 0]
        
        def is_valid(x, y, z):
            return 0 <= x < n and 0 <= y < n and 0 <= z < n
    
        def dfs(x, y, z, cube):
            directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
            # random the directions
            np.random.shuffle(directions)
            if cube.sum() >= number_of_blocks:
                return
            cube[x, y, z] = 1
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                if is_valid(nx, ny, nz) and cube[nx, ny, nz] == 0:
                    dfs(nx, ny, nz, cube)
        # 从起始点开始DFS
        dfs(start_x, start_y, start_z, cube_xyz_idx)
        return cube_xyz_idx
    
    def generate_spherical_surface(self, r):
        n = self.env.rubik_x_size
        center = (n // 2, n // 2, n // 2)
        cube_xyz_idx = np.zeros((n, n, n), dtype=int)
        # r = n // 2

        for x in range(n):
            for y in range(n):
                for z in range(n):
                    # Calculate the distance from the current point to the sphere center
                    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

                    # Check if the distance is within the spherical shell range
                    if r - 0.5 <= distance < r + 0.5:
                        cube_xyz_idx[x, y, z] = 1
        
        return cube_xyz_idx
    
    def generate_perlin_noise(self, size):
        scale, threshold = 0.1, 0.2
        cubes = []
        n = self.env.rubik_x_size
        cube_xyz_idx = np.zeros((n, n, n), dtype=int)
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    noise_value = pnoise3(x * scale, y * scale, z * scale)
                    if noise_value > threshold:
                        cubes.append((x, y, z))
            
        for x, y, z in cubes:
            cube_xyz_idx[x, y, z] = 1
        
        return cube_xyz_idx
        
import robosuite as suite
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
def create_env(task="connected_cube"):
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
            # "top2bottom", "bottom2top",
            # "sideview_0", "sideview_45", "sideview_90", "sideview_135",
            # "sideview_180", "sideview_225", "sideview_270", "sideview_315",
            "frontview", "topview", "sideview",
        ], 
        "camera_depths": True,
        "camera_heights": 512, # 1024
        "camera_widths": 512, # 1024
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
        task=task,
        # task="connected_cube", # see available_tasks in SpatialIntelligenceWrapper
        # task="spherical_surface",
        # task="perlin_noise",
    )
    return env