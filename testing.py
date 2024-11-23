import numpy as np
import time
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

def reset():
    grid = np.random.randint(0, 4 + 1, size=(4,4))
    return grid

def main():

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


if __name__ == "__main__":
    main()