# import json
# # import pdb
# import numpy as np

# import os
# import time, datetime
# import gymnasium as gym
# from OpenGL import GL

# from utils import traverse_grid_3d, save_video, search_for_place_cube_actions

# from spatial_intelligence_wrapper import create_env
# if __name__ == "__main__": 
#     # test the sim speed of robosuite

#     def ignore_gl_errors(*args, **kwargs):
#         pass
#     GL.glCheckError = ignore_gl_errors
#     choose_robosuite = True
#     save_video_flag = True
    
#     env = create_env()
#     eps = 1
#     i_eps = 0
#     step_count = 0
#     start_time = time.time()
#     images = []
#     front_images = []
#     agent_images = []
#     robot_images = []
    
#     env.reset()
    
#     view_name = "frontview" # sideview, frontview, agentview, robot0_eye_in_hand, birdview
#     while i_eps < eps:
#         obs = env.reset()
#         front_images.append(np.flipud(obs['obs'][view_name]))
#         ep_step_count = 0
#         for action in search_for_place_cube_actions(
#             env.env.rubik_x_size, env.env.rubik_y_size, env.env.rubik_y_size, 
#             env.cube_xyz_idx, env.env.rubik_red_cube_xyz_idx
#         ):
#             result, r, d, _ = env.step(action)
#             obs = result["obs"]
#             front_images.append(np.flipud(obs[view_name]))
#             print(f"result: {result['success']}, message: {result['message']}")
#             if "direction cannot be placed because there is no cube below it." in result['message']:
#                 raise NotImplementedError(f"action: {action}, result: {result['message']}")
#             step_count += 1
#         print(f"eps: {i_eps}, steps: {step_count}, blocks success: {r}")
#         i_eps += 1
#     end_time = time.time()
    
#     total_time = end_time - start_time
#     time_per_step = total_time / step_count
#     # print(f"time_per_step: {time_per_step}")
    
#     # save video 
#     if save_video_flag:
#         print(f"save video...")
#         os.makedirs("./video", exist_ok=True)
#         images.extend(front_images)
#         save_video(f"./video/SpatialIntelligence_{view_name}.mp4", [images])


import numpy as np
# from utils import find_path

from collections import deque

def find_path(grid, start, end):
    if isinstance(grid, np.ndarray):
        grid = grid.tolist()
    if isinstance(start, np.ndarray):
        start = tuple(start.tolist())
    if isinstance(start, list):
        start = tuple(start)
    if isinstance(end, np.ndarray):
        end = tuple(end.tolist())
    if isinstance(end, list):
        end = tuple(end)
    
    # 检查grid是否为空
    if not grid or not grid[0]:
        return []
    
    rows = len(grid)
    cols = len(grid[0])
    
    # 验证start和end的坐标是否合法，并且值为1
    sx, sy = start
    ex, ey = end
    if not (0 <= sx < rows and 0 <= sy < cols) or grid[sx][sy] != 1:
        return []
    if not (0 <= ex < rows and 0 <= ey < cols) or grid[ex][ey] != 1:
        return []
    
    # 处理起点和终点相同的情况
    if start == end:
        return [start]
    
    # 初始化队列和访问集合，记录父节点
    queue = deque()
    queue.append(start)
    visited = set()
    visited.add(start)
    parent = {}
    parent[start] = None
    
    # 四个移动方向：上下左右
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    found = False
    while queue:
        current = queue.popleft()
        # 到达终点
        if current == end:
            found = True
            break
        # 探索四个方向
        for dx, dy in directions:
            nx = current[0] + dx
            ny = current[1] + dy
            # 检查是否在网格范围内
            if 0 <= nx < rows and 0 <= ny < cols:
                # 检查是否为可通过的路径且未被访问过
                if grid[nx][ny] == 1 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = current
                    queue.append((nx, ny))
    
    # 如果未找到路径，返回空列表
    if not found:
        return []
    
    # 回溯构造路径
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = parent[node]
    # 反转路径，从起点到终点
    path.reverse()
    return path

grid = np.array(
    [
        [1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
)

# start = (1, 3)
# end = (1, 0)
start = [2, 0]
end = [0, 1]

path = find_path(grid, start, end)
print(path)
print(not path)
