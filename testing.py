import numpy as np
import time
import timeit
from collections import deque
def vm_new(grid):
    return list(zip(*np.nonzero(grid)))

def vm_old(grid):

    valid_actions = []
    rows, cols = grid.shape
    for row in range(rows):
        for col in range(cols):
            symbol = grid[row, col]
            if symbol != 0:
                valid_actions.append((row, col))
    return valid_actions  

def get_valid_actions(grid):
        return list(zip(*np.nonzero(grid)))

def find_connected_blocks_new(grid, row, col, symbol):
    visited = set()  # Use a set for fast lookups
    to_visit = [(row, col)]  # Use a list for DFS-like traversal
    connected = []

    while to_visit:
        r, c = to_visit.pop()  # Pop from the end (DFS-like)
        if (r, c) in visited:
            continue
        if grid[r, c] != symbol:
            continue

        visited.add((r, c))
        connected.append((r, c))

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and (nr, nc) not in visited:
                to_visit.append((nr, nc))

    return connected

def find_connected_blocks(grid, row, col, symbol):
    visited = set()
    to_visit = [(row, col)]
    while to_visit:
        r, c = to_visit.pop()
        if (r, c) in visited:
            continue
        if grid[r, c] != symbol:
            continue
        visited.add((r, c))

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if grid[nr, nc] == symbol:
                    to_visit.append((nr, nc))
    return visited

def find_unique_moves_old(grid):
    moves = get_valid_actions(grid)

    for m in moves:
        if m in moves:
            alike_moves = find_connected_blocks(grid, m[0], m[1], grid[m[0], m[1]])
            alike_moves.remove(m)

            moves = [move for move in moves if move not in alike_moves]

    return moves

def find_unique_moves_new(grid):
    
    moves = set(get_valid_actions(grid))  # Use a set for faster lookups
    unique_moves = set()  # Store unique moves
    processed = set()  # Track processed moves

    for move in moves:
        if move not in processed:
            alike_moves = set(find_connected_blocks_new(grid, move[0], move[1], grid[move[0], move[1]]))
            processed.update(alike_moves)  # Mark all alike moves as processed
            unique_moves.add(move)  # Only add the original move to unique moves

    return list(unique_moves)  # Convert back to list if needed

def reset():
    grid = np.random.randint(0, 4 + 1, size=(7,9))
    return grid

def main():
    grid = reset()
    
    """
    print(grid)

    test = set(find_unique_moves_new(grid))
    u_m = set(find_unique_moves_old(grid))

    diff = test ^ u_m

    print(diff)
    """




    print(timeit.timeit(lambda:find_unique_moves_new(grid), number = 10*7))
    print(timeit.timeit(lambda:find_unique_moves_old(grid), number = 10*7))
    """
    grids = [reset() for i in range(10**7)]
    grid = reset()

    start_new = time.time()
    for grid in grids:
        vm_new(grid)
    end_new = time.time() - start_new
   
    start_old = time.time()

    for grid in grids:
        vm_old(grid)
    end_old = time.time() - start_old

    print(f"New function time:{end_new}, old function time: {end_old}")
    """

if __name__ == "__main__":
    main()