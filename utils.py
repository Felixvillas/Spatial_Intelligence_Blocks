import numpy as np

def traverse_grid_2d(grid, start):
    """
    Traverses a connected subgraph in the grid starting from the given start position.

    Parameters:
        grid (np.ndarray): The 2D grid world represented as a numpy array.
        start (tuple): The starting position (row, col).

    Returns:
        list: A list of positions representing the traversal path.
    """
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)  # Keep track of visited cells
    path = []

    def is_valid(x, y):
        """Check if the position (x, y) is valid and unvisited in the grid."""
        return 0 <= x < rows and 0 <= y < cols and grid[x, y] == 1 and not visited[x, y]

    def dfs(x, y):
        """Recursive DFS function to traverse the grid."""
        visited[x, y] = True
        path.append((x, y))

        # Define movement directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny):
                dfs(nx, ny)
                path.append((x, y))  # Backtrack to current cell

    # Start the DFS from the initial position
    dfs(*start)
    return path

# # Example usage
# grid = np.array([
#     [1, 1, 0],
#     [1, 0, 0],
#     [1, 1, 1]
# ])
# start_position = (0, 0)

# path = traverse_grid_2d(grid, start_position)
# print("Traversal Path:", path)


# import numpy as np

def find_path(grid, start, end):
    """
    Finds a path between two points in a connected subgraph of the 2D grid.

    Parameters:
        grid (np.ndarray): The 2D grid world represented as a numpy array.
        start (tuple): The starting position (x, y).
        end (tuple): The ending position (x, y).

    Returns:
        list: A list of positions representing the path from start to end, or an empty list if no path exists.
    """
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)  # Keep track of visited cells
    path = []

    def is_valid(x, y):
        """Check if the position (x, y) is valid and unvisited in the grid."""
        return 0 <= x < rows and 0 <= y < cols and grid[x, y] == 1 and not visited[x, y]

    def dfs(x, y):
        """Recursive DFS function to find the path."""
        if (x, y) == end:
            path.append((x, y))
            return True

        visited[x, y] = True
        path.append((x, y))

        # Define movement directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny):
                if dfs(nx, ny):
                    return True

        # Backtrack if no path is found
        path.pop()
        return False

    # Start DFS from the initial position
    if dfs(*start):
        return path
    else:
        return []

# # Example usage
# grid = np.array([
#     [1, 1, 0],
#     [1, 0, 0],
#     [1, 1, 0]
# ])
# start_position = (0, 0)
# end_position = (0, 0)

# path = find_path(grid, start_position, end_position)
# print("Path from start to end:", path)



# import numpy as np
################## Test use, not guaranteed to be completely correct ##################

def find_not_traversed(grid, traverse_nodes):
    grid_copy = grid.copy()
    for x, y in traverse_nodes:
        grid_copy[x, y] = 0
    return np.argwhere(grid_copy).tolist()

def traverse_grid_3d(grid, start):
    """
    Traverses a connected subgraph in the grid starting from the given start position,
    ensuring traversal is performed layer by layer along the z-axis.

    Parameters:
        grid (np.ndarray): The 3D grid world represented as a numpy array.
        start (tuple): The starting position (x, y, z).

    Returns:
        list: A list of positions representing the traversal path.
    """
    dims = grid.shape
    path = []

    # Traverse each layer along the z-axis from bottom to top
    current_start = start
    for z in range(dims[2]):
        if grid[:, :, z].sum() == 0:
            continue # or break
        z_path = traverse_grid_2d(grid[:, :, z], current_start[:2])
        z_path = [(x, y, z) for x, y in z_path]  # Convert 2D path to 3D path
        # while True:
        #     no_travsered = find_not_traversed(grid[:, :, z], [item[:2] for item in z_path])
        #     if not no_travsered:
        #         break
        #     for no_travsered_node in no_travsered:
        #         # down a layer
        #         z_path.append((z_path[-1][0], z_path[-1][1], z - 1))
        #         # last node in z_path
        #         last_node = z_path[-1]
        #         inter_layer_path = find_path(grid[:, :, z - 1], last_node[:2], no_travsered_node[:2])
        #         inter_layer_path = inter_layer_path[1:]
        #         inter_layer_path = [(x, y, z - 1) for x, y in inter_layer_path]
        #         z_path.extend(inter_layer_path)
        #         z_path.append([no_travsered_node[0], no_travsered_node[1], z])
        
        
        path.extend(z_path)
        
        # 找到该层和上层连通的点，通过数组与运算找到
        # Find the connected cells between the current and next layer
        if z < dims[2] - 1:
            if grid[:, :, z].sum() == 0 or grid[:, :, z + 1].sum() == 0:
                continue
            connected_cells = np.logical_and(grid[:, :, z], grid[:, :, z + 1])
            connected_cells = np.argwhere(connected_cells)
            connected_cell = connected_cells[0]
            next_start = (connected_cell[0], connected_cell[1], z + 1)
            # 找到从current_start[:2]到next_start[:2]的路径
            # Find the path from current_start[:2] to next_start[:2]
            inter_layer_path = find_path(grid[:, :, z], current_start[:2], next_start[:2])
            # print(f"inter_layer_path: {inter_layer_path}")
            # 因为traverse_grid_2d会回到起始点，所以要去掉inter_layer_path的第一个点
            # Since traverse_grid_2d will return to the starting point, we need to remove the first point of inter_layer_path
            inter_layer_path = inter_layer_path[1:]
            inter_layer_path = [(x, y, z) for x, y in inter_layer_path]
            path.extend(inter_layer_path)
            current_start = next_start
            
        

    return path

import imageio
def save_video(video_path_name, datas, fps=20):
    if not video_path_name.endswith(".mp4"):
        video_path_name += ".mp4"
    
    video_writer = imageio.get_writer(video_path_name, fps=fps)
    for data in datas:
        for img in data:
            video_writer.append_data(img[::-1])
            
    video_writer.close()
    

from collections import deque

def find_path_v1(grid, start, end):
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

from copy import deepcopy

import json
def search_for_think_and_answer_v1(rubik_x_size, rubik_y_size, rubik_z_size, target_cube_xyz_idx, red_cube_xyz_idx):
    
    def think(t, todo_place_actions, todo_move_actions, actual_action):
        if t == 1:
            place_actions_str = ", ".join([f"{item['direction']}" for item in todo_place_actions])
            think_str = f"Building block build priority: from backward to forward, from left to right, from down to up. "
            think_str += f"The current building block has one less block at the <{place_actions_str}> positions adjacent to the red cursor than the target building block. "
            think_str += f"According to the building block build priority, we need to place a block at the <{todo_place_actions[0]['direction']}> position adjacent to the red cursor of the current building block."
        
        elif t == 2:
            move_actions_str = ", ".join([f"{item['direction']}" for item in todo_move_actions])
            think_str = "Building block build priority: from backward to forward, from left to right, from down to up. "
            think_str += "The blocks adjacent to the current building block in the red cursor are the same as the target building block, but there are still unbuilt blocks in the layer where the red cursor is located. "
            think_str += "According to the building block build priority, blocks that are as backward and left as possible and adjacent to the already built blocks should be built first. "
            think_str += f"So we move the red cursor to the position of the block adjacent to the block to be built in the order of <{move_actions_str}>, and then place the block at the <{todo_place_actions[0]['direction']}> position."
            
        elif t == 3:
            move_actions_str = ", ".join([f"{item['direction']}" for item in todo_move_actions])
            think_str = "Building block build priority: from backward to forward, from left to right, from down to up. "
            think_str += "The blocks adjacent to the current building block in the red cursor are the same as the target building block, and the layer where the red cursor is built has been completed. "
            think_str += "There are still unfinished layers above the red cursor. "
            think_str += "Priority should be given to building layers that are as down as possible and adjacent to the already built layer. "
            think_str += f"So we first move the red cursor to the block adjacent to the layer to be built in the order of <{move_actions_str}>, and then place the block at the <up> position."
        else:
            raise NotImplementedError(f"Unknown think type: {t}")
        
        think_str = "<think>" + think_str + "</think>\n"
        # json_str = "```json" + json.dumps(actual_action) + "```"
        # think_str += "<answer>" + json_str + "</answer>"
        think_str += "<answer>" + json.dumps(actual_action) + "</answer>"
        
        return think_str
    
    def valid_pos(pos):
        x, y, z = pos
        return 0 <= x < rubik_x_size and 0 <= y < rubik_y_size and 0 <= z < rubik_z_size
    
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
        else:
            raise NotImplementedError(f"Unknown delta_str: {delta_str}")
        return direction
    
    directions = [
        {
            "direction": "backward",
            "delta": np.array([-1, 0, 0]),
        },
        {
            "direction": "forward",
            "delta": np.array([1, 0, 0]),
        },
        {
            "direction": "left",
            "delta": np.array([0, -1, 0]),
        },
        {
            "direction": "right",
            "delta": np.array([0, 1, 0]),
        },
        # {
        #     "direction": "up",
        #     "delta": np.array([0, 0, 1]),
        # },
        # {
        #     "direction": "down",
        #     "delta": np.array([0, 0, -1]),
        # }
    ]
    actions = []
    thinks = []
    # start pos is 0, 0, 0
    start_cube_xyz_idx = np.array([0, 0, 0])
    
    assert np.all(start_cube_xyz_idx == red_cube_xyz_idx), f"start_cube_xyz_idx: {start_cube_xyz_idx}, red_cube_xyz_idx: {red_cube_xyz_idx} should be the same"

    current_cube_xyz_idx = np.zeros_like(target_cube_xyz_idx)
    current_cube_xyz_idx[start_cube_xyz_idx[0], start_cube_xyz_idx[1], start_cube_xyz_idx[2]] = 1
    for z in range(rubik_z_size):
        while np.any(target_cube_xyz_idx[:, :, z] - current_cube_xyz_idx[:, :, z]):
            for item in directions:
                place_block_flag = False
                actions_temp = []
                delta_pos = red_cube_xyz_idx + item["delta"]
                if not valid_pos(delta_pos):
                    continue
                if target_cube_xyz_idx[delta_pos[0], delta_pos[1], delta_pos[2]] and not current_cube_xyz_idx[delta_pos[0], delta_pos[1], delta_pos[2]]:
                    actions_temp.append(
                        {
                            "action": "place_block",
                            "direction": item["direction"],
                        }
                    )
                    
            for item in actions_temp:
                actions.append(
                    {
                        "action": "place_block",
                        "direction": item["direction"],
                    }
                )
                print(f"1. place block at: {item['direction']}")
                current_cube_xyz_idx[delta_pos[0], delta_pos[1], delta_pos[2]] = 1
                red_cube_xyz_idx = delta_pos
                place_block_flag = True
                break
            if place_block_flag:
                assert actions[-1] == actions_temp[0]
                think_str = think(1, actions_temp, [], actual_action=actions[-1])
                thinks.append(think_str)
            if not place_block_flag:
                if np.all((target_cube_xyz_idx[:, :, z] - current_cube_xyz_idx[:, :, z]) == 0):
                    break
                # find the most backward and most left no-placed block
                for y in range(rubik_y_size):
                    for x in range(rubik_x_size):
                        find_path_flag = False
                        if target_cube_xyz_idx[x, y, z] and not current_cube_xyz_idx[x, y, z]:
                            block = np.array([x, y, z])
                            # find the path from red_cube_xyz_idx to block in current_cube_xyz_idx
                            current_cube_xyz_idx_copy = deepcopy(current_cube_xyz_idx)
                            current_cube_xyz_idx_copy[block[0], block[1], block[2]] = 1
                            path = find_path_v1(current_cube_xyz_idx_copy[:, :, z], red_cube_xyz_idx[:2], block[:2])
                            if not path:
                                continue
                            find_path_flag = True
                            path = path[1:] # remove the first node
                            break
                    if find_path_flag:
                        break
                
                if find_path_flag:
                    start_actions_len = len(actions)
                    for p in path[:-1]:
                        del_pos = [p[0] - red_cube_xyz_idx[0], p[1] - red_cube_xyz_idx[1]]
                        del_pos_str = f"{del_pos[0]}_{del_pos[1]}_{0}"
                        direction = delta_str2direction(del_pos_str)
                        actions.append(
                            {
                                "action": "move_cursor",
                                "direction": direction,
                            }
                        )
                        print(f"2. move cursor to: {direction}")
                        red_cube_xyz_idx = np.array([p[0], p[1], z])
                    end_actions_len = len(actions)
                    # place the block
                    del_pos = [path[-1][0] - red_cube_xyz_idx[0], path[-1][1] - red_cube_xyz_idx[1]]
                    del_pos_str = f"{del_pos[0]}_{del_pos[1]}_{0}"
                    direction = delta_str2direction(del_pos_str)
                    actions.append(
                        {
                            "action": "place_block",
                            "direction": direction,
                        }
                    )
                    print(f"3. place block at: {direction}")
                    current_cube_xyz_idx[path[-1][0], path[-1][1], z] = 1
                    red_cube_xyz_idx = np.array([path[-1][0], path[-1][1], z])
                    
                    for i in range(start_actions_len, end_actions_len):
                        think_str = think(2, [actions[-1]], actions[i:end_actions_len], actual_action=actions[i])
                        thinks.append(think_str)
                    
                    think_str = think(1, [actions[-1]], [], actual_action=actions[-1]) # maybe need modify as no detect on the adjacent blocks
                    thinks.append(think_str)
                else:
                    # need find path from the down or downdown... layer
                    for y in range(rubik_y_size):
                        for x in range(rubik_x_size):
                            find_path_flag = False
                            if target_cube_xyz_idx[x, y, z] and not current_cube_xyz_idx[x, y, z]:
                                block = np.array([x, y, z])
                                for z_ in range(z - 1, -1, -1):
                                    find_path_flag = False
                                    path = find_path_v1(current_cube_xyz_idx[:, :, z_], red_cube_xyz_idx[:2], block[:2])
                                    if not path:
                                        print(f"current_cube_xyz_idx[:, :, {z_}]: {current_cube_xyz_idx[:, :, z_]}, red_cube_xyz_idx: {red_cube_xyz_idx}, block: {block}")
                                        continue
                                    find_path_flag = True
                                    path = [[p[0], p[1], z_] for p in path[1:-1]]
                                    _path = [[red_cube_xyz_idx[0], red_cube_xyz_idx[1], z__] for z__ in range(z - 1, z_ - 1, -1)]
                                    _path.extend(path)
                                    path = deepcopy(_path)
                                    del _path
                                    _path = [[block[0], block[1], z__] for z__ in range(z_, z + 1)]
                                    path.extend(_path)
                                    path = deepcopy(path)
                                    del _path
                                    break
                            if find_path_flag:
                                break
                        if find_path_flag:
                            break
                    if not find_path_flag:
                        print(f"red_cube_xyz_idx: {red_cube_xyz_idx}, block: {block}")
                        for z_ in range(z - 1, -1, -1):
                            print(f"current_cube_xyz_idx[:, :, {z_}]: {current_cube_xyz_idx[:, :, z_]}")
                        raise NotImplementedError(f"not find path from the down or downdown... layer")
                    
                    start_actions_len = len(actions)
                    for p in path[:-1]:
                        # try:
                        del_pos = [p[0] - red_cube_xyz_idx[0], p[1] - red_cube_xyz_idx[1], p[2] - red_cube_xyz_idx[2]]
                        del_pos_str = f"{del_pos[0]}_{del_pos[1]}_{del_pos[2]}"
                        direction = delta_str2direction(del_pos_str)
                        # except Exception as e:
                        #     print(f"path: {path}, \n red_cube_xyz_idx: {red_cube_xyz_idx}, \n block: {block}")
                        #     raise e
                        actions.append(
                            {
                                "action": "move_cursor",
                                "direction": direction,
                            }
                        )
                        print(f"4. move cursor to: {direction}")
                        red_cube_xyz_idx = np.array([p[0], p[1], p[2]])
                    end_actions_len = len(actions)
                    
                    # place the block in up
                    actions.append(
                        {
                            "action": "place_block",
                            "direction": "up",
                        }
                    )
                    print(f"5. place block at: up")
                    current_cube_xyz_idx[block[0], block[1], z] = 1
                    red_cube_xyz_idx = np.array([block[0], block[1], z])
                    
                    
                    for i in range(start_actions_len, end_actions_len):
                        think_str = think(2, [actions[-1]], actions[i:end_actions_len], actual_action=actions[i])
                        thinks.append(think_str)
                    
                    think_str = think(1, [actions[-1]], [], actual_action=actions[-1]) # maybe need modify as no detect on the adjacent blocks
                    thinks.append(think_str)
        
        if z != rubik_z_size - 1 and np.any(target_cube_xyz_idx[:, :, z + 1] - current_cube_xyz_idx[:, :, z + 1]):
            # find the most backward and most left no-placed block in layer z + 1
            for y in range(rubik_y_size):
                for x in range(rubik_x_size):
                    find_block = False
                    if target_cube_xyz_idx[x, y, z + 1] and not current_cube_xyz_idx[x, y, z + 1]:
                        block = np.array([x, y, z + 1])
                        find_block = True
                        break
                if find_block:
                    break
                    
            for z_ in range(z, -1, -1):
                find_path_flag = False
                path = find_path_v1(current_cube_xyz_idx[:, :, z_], red_cube_xyz_idx[:2], block[:2])
                if not path:
                    continue
                find_path_flag = True
                path = [[p[0], p[1], z_] for p in path[1:-1]]
                _path = [[red_cube_xyz_idx[0], red_cube_xyz_idx[1], z__] for z__ in range(z - 1, z_ - 1, -1)]
                _path.extend(path)
                path = deepcopy(_path)
                del _path
                _path = [[block[0], block[1], z__] for z__ in range(z_, z + 1)]
                path.extend(_path)
                path = deepcopy(path)
                del _path
                # print(f"path: {path}, red_cube_xyz_idx: {red_cube_xyz_idx}, block: {block}")
                break
            if not find_path_flag:
                raise NotImplementedError(f"not find path from the down or downdown... layer")
            
            start_actions_len = len(actions)
            for p in path:
                # try:
                del_pos = [p[0] - red_cube_xyz_idx[0], p[1] - red_cube_xyz_idx[1], p[2] - red_cube_xyz_idx[2]]
                del_pos_str = f"{del_pos[0]}_{del_pos[1]}_{del_pos[2]}"
                if del_pos_str == "0_0_0":
                    continue
                direction = delta_str2direction(del_pos_str)
                actions.append(
                    {
                        "action": "move_cursor",
                        "direction": direction,
                    }
                )
                print(f"4. move cursor to: {direction}")
                red_cube_xyz_idx = np.array([p[0], p[1], p[2]])
            end_actions_len = len(actions)
            
            # place the block in up
            actions.append(
                {
                    "action": "place_block",
                    "direction": "up",
                }
            )
            print(f"7. place block at: up")
            current_cube_xyz_idx[block[0], block[1], z + 1] = 1
            red_cube_xyz_idx = np.array([block[0], block[1], z + 1])
            
            for i in range(start_actions_len, end_actions_len):
                think_str = think(3, [actions[-1]], actions[i:end_actions_len], actual_action=actions[i])
                thinks.append(think_str)
            
            think_str = think(1, [actions[-1]], [], actual_action=actions[-1]) # maybe need modify as no detect on the adjacent blocks
            thinks.append(think_str)
    
    assert len(actions) == len(thinks)
    return actions, thinks


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