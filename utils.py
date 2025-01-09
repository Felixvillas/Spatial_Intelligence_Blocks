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

# # Example usage
# grid = np.array([
#     [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
#     [[1, 1, 1], [0, 0, 1], [0, 0, 0]],
#     [[1, 0, 0], [1, 1, 1], [0, 0, 1]]
# ])
# start_position = (0, 0, 0)

# path = traverse_grid_3d(grid, start_position)
# print("Traversal Path:", path)

# print("length of set of path:", len(set(path)))
# assert len(set(path)) == grid.sum()  # Check if all cells are visited
# breakpoint()  # Debugging

def search_for_place_cube_actions(rubik_x_size, rubik_y_size, rubik_z_size, target_cube_xyz_idx, red_cube_xyz_idx):
    actions = []
    # find the start pos
    start_cube_xyz_idx = None
    for z in range(rubik_z_size):
        for x in range(rubik_x_size - 1, -1, -1):
            for y in range(rubik_y_size):
                if target_cube_xyz_idx[x, y, z] == 1:
                    start_cube_xyz_idx = np.array([x, y, z])
                    break
            if start_cube_xyz_idx is not None:
                break
        if start_cube_xyz_idx is not None:
            break
    
    assert np.all(start_cube_xyz_idx == red_cube_xyz_idx), f"start_cube_xyz_idx: {start_cube_xyz_idx}, red_cube_xyz_idx: {red_cube_xyz_idx} should be the same"
    
    # find the path
    path = traverse_grid_3d(target_cube_xyz_idx, tuple(start_cube_xyz_idx))
    # print(f"path: {path}")
    
    placed_cube_xyz_idx = np.zeros_like(target_cube_xyz_idx)
    placed_cube_xyz_idx[start_cube_xyz_idx[0], start_cube_xyz_idx[1], start_cube_xyz_idx[2]] = 1
    # breakpoint()
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        delta_node = [next_node[0] - current_node[0], next_node[1] - current_node[1], next_node[2] - current_node[2]]
        delta_node_str = f"{delta_node[0]}_{delta_node[1]}_{delta_node[2]}"
        # direction
        if delta_node_str == "1_0_0":
            direction = "forward"
        elif delta_node_str == "-1_0_0":
            direction = "backward"
        elif delta_node_str == "0_1_0":
            direction = "right"
        elif delta_node_str == "0_-1_0":
            direction = "left"
        elif delta_node_str == "0_0_1":
            direction = "up"
        elif delta_node_str == "0_0_-1":
            direction = "down"
        else:
            # breakpoint()
            # raise NotImplementedError(f"Unknown delta_node_str: {delta_node_str}, Current node: {current_node}, Next node: {next_node}")
            pass
        
        # action
        if not placed_cube_xyz_idx[next_node[0], next_node[1], next_node[2]]:
            action = "place_block"
            placed_cube_xyz_idx[next_node[0], next_node[1], next_node[2]] = 1
        else:
            action = "move_cursor"
        actions.append(
            {
                "action": action,
                "direction": direction,
            }
        )
    
    return actions

import imageio
def save_video(video_path_name, datas, fps=20):
    if not video_path_name.endswith(".mp4"):
        video_path_name += ".mp4"
    
    video_writer = imageio.get_writer(video_path_name, fps=fps)
    for data in datas:
        for img in data:
            video_writer.append_data(img[::-1])
            
    video_writer.close()