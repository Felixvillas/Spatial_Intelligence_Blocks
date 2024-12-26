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
        # print(f"z_path: {z_path}")
        z_path = [(x, y, z) for x, y in z_path]  # Convert 2D path to 3D path
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

import imageio
def save_video(video_path_name, datas, fps=20):
    if not video_path_name.endswith(".mp4"):
        video_path_name += ".mp4"
    
    video_writer = imageio.get_writer(video_path_name, fps=fps)
    for data in datas:
        for img in data:
            video_writer.append_data(img[::-1])
            
    video_writer.close()