import numpy as np
import random
from PIL import Image
import sys, os
import matplotlib.pyplot as plt
from copy import deepcopy

from robosuite.environments.manipulation.spatial_intelligence import SpatialIntelligence

class SpatialIntelligenceWrapper:
    def __init__(
        self,
        env,
    ):
        assert isinstance(env, SpatialIntelligence)
        self.env = env
        self.views = deepcopy(env.camera_names)
        self.operate_view = "top2bottom"
        self.reset_vars()
        self.episode_max_len = 200
        
        # self.generate_task()
        # sys.exit("test")
        
        
    def reset_vars(self):
        self.perspective_view = "top2bottom"
        self.blocks_views = {
            key: None for key in self.views
        }
        self.episode_step = 0
        
    def reset(self):
        self.reset_vars()
        self.generate_task()
        obs = self.env.reset()
        return {
            "perspective_view": obs[f"{self.perspective_view}_image"],
            "operate_view": obs[f"{self.operate_view}_image"],
        }
    
    def step(self, action):
        act, direction = action["action"], action["direction"]
        if act == "view":
            if direction not in self.views:
                results = {
                    "obs": {
                        "perspective_view": obs[f"{self.perspective_view}_image"],
                        "operate_view": obs[f"{self.operate_view}_image"],
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
        elif act == "move":
            results = self.env.move_red_cube(direction)
        elif act == "place":
            results = self.env.place_cube(direction, self.cube_xyz_idx)
        else:
            raise NotImplementedError(f"Unknown action: {act}")
        
        obs, r, d, _ = self.env.step(np.zeros([]))
        self.episode_step += 1
        results["obs"] = {
            "perspective_view": obs[f"{self.perspective_view}_image"],
            "operate_view": obs[f"{self.operate_view}_image"],
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
        return results, r, d, {}
    
    def check_success(self):
        if np.all(self.env.rubik_xyz_idx_exists == self.cube_xyz_idx):
            return True
        return False

    def generate_rubik_by_cube_xyz_idx(self, cube_xyz_idx):
        # generate rubik by referring to the cube
        self.env.reset()
        self.env.generate_rubik_by_cube_xyz_idx(cube_xyz_idx)
        obs, r, d, _ = self.env.step(np.zeros([]))
        return obs

    def generate_connected_cube(self, number_of_blocks):
        # 初始化魔方矩阵，0代表空，1代表积木块
        n = self.env.rubik_x_size
        cube = np.zeros((n, n, n), dtype=int)
        # 随机选择一个起始点
        start_x, start_y, start_z = [0, 0, 0]
        
        def is_valid(x, y, z):
            return 0 <= x < n and 0 <= y < n and 0 <= z < n
    
        def dfs(x, y, z, cube):
            directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
            if cube.sum() >= number_of_blocks:
                return
            cube[x, y, z] = 1
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                if is_valid(nx, ny, nz) and cube[nx, ny, nz] == 0:
                    dfs(nx, ny, nz, cube)
        # 从起始点开始DFS
        dfs(start_x, start_y, start_z, cube)
        return cube
        
    def generate_task(self):
        cube_xyz_idx = self.generate_connected_cube(
            np.random.randint(
                low=self.env.rubik_x_size + 1, 
                high=self.env.rubik_x_size * self.env.rubik_y_size * self.env.rubik_z_size + 1
            )
            # 8, # for debug
        )
        self.cube_xyz_idx = cube_xyz_idx
        obs = self.generate_rubik_by_cube_xyz_idx(cube_xyz_idx)
        for view in self.views:
            # # save image: obs[f"{view}_image"] as png by Image.fromarray
            # img = Image.fromarray(obs[f"{view}_image"])
            # # Flip the img upside down and then save it
            # img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # img.save(f"task_view/{view}.png")
            self.blocks_views[view] = np.flipud(obs[f"{view}_image"])
            os.makedirs("task_view", exist_ok=True)
            plt.imsave(f"task_view/{view}.png", self.blocks_views[view])
            
                
                
    def task_prompt(self):
        task_instruction = """
Here are several perspective views of a complete block:
XXX
Your task is to assemble the block based on these perspective views.
You have the perspective view aaa and the operation perspective bbb. Under your viewing perspective, the blocks that have been built are as follows:
XXX
Under your operational perspective, the blocks that have been built are as follows:
XXX
The red square represents the cursor, and the operations you can use include the following three categories:
1. Switch perspectives: Available perspectives include {"frontview", "birdview", "sideview"}
2. Move the cursor: Move the cursor within the built blocks, available directions include {"up", "down", "left", "right", "forward", "backward"}
3. Place blocks: Place blocks on a face of the cursor, available faces/directions include {"up", "down", "left", "right", "forward", "backward"}        
        """
        return task_instruction
        