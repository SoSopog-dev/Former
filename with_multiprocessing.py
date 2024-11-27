import numpy as np
import time
from multiprocessing import Pool, Manager


class FormerGame:
    def __init__(self,grid, grid_size, num_symbols):
        self.grid_size = grid_size
        self.num_symbols = num_symbols
        self.reset(grid)

    def reset(self, grid):
        self.grid = grid.copy()
        return self.get_state()
    
    def get_state(self):
        return self.grid.copy()

    def get_valid_actions(self):
        return list(zip(*np.nonzero(self.grid)))

    def step(self, action):
        row, col = action
        symbol = self.grid[row, col]
        if symbol == 0:
            return self.get_state(), 0, False

        connected_blocks = self.find_connected_blocks(row, col, symbol)

        for r, c in connected_blocks:
            self.grid[r, c] = 0

        self.apply_gravity()

        done = np.all(self.grid == 0)

        return self.get_state(), done

    def find_connected_blocks(self, row, col, symbol):
        visited = set()
        to_visit = [(row, col)]
        while to_visit:
            r, c = to_visit.pop()
            if (r, c) in visited:
                continue
            if self.grid[r, c] != symbol:
                continue
            visited.add((r, c))

            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1]:
                    if self.grid[nr, nc] == symbol:
                        to_visit.append((nr, nc))
        return visited

    def apply_gravity(self):
        rows, cols = self.grid.shape
        for col in range(cols):
            column = self.grid[:, col]
            non_zero = column[column != 0]
            zeros = np.zeros(rows - len(non_zero), dtype=int)
            self.grid[:, col] = np.concatenate((zeros, non_zero))

def find_unique_moves(game):
    """
    moves = set(game.get_valid_actions())  # Use a set for faster lookups
    unique_moves = set()  # Store unique moves
    processed = set()  # Track processed moves

    for move in moves:
        if move not in processed:
            alike_moves = game.find_connected_blocks(move[0], move[1], game.grid[move[0], move[1]])
            processed.update(alike_moves)  # Mark all alike moves as processed
            unique_moves.add(move)  # Only add the original move to unique moves

    return list(unique_moves)  # Convert back to list if needed
    """
    #can be optimized
    moves = game.get_valid_actions()

    for m in moves:
        if m in moves:
            alike_moves = game.find_connected_blocks(m[0], m[1], game.grid[m[0], m[1]])
            alike_moves.remove(m)

            moves = [move for move in moves if move not in alike_moves]

    return moves
    

def solve_parallel(args):
    """Helper function for multiprocessing. Solves a sub-problem."""
    state, game, moves, best_solution, len_best_solution, depth = args
    return solve(state, game, moves, best_solution, len_best_solution, depth)

def solve(state, game, moves, best_solution, len_best_solution, depth):
    if depth >= len_best_solution:
        return best_solution, len_best_solution

    unique_moves = find_unique_moves(game)

    # Prepare for multiprocessing
    if depth == 0:  # Only use multiprocessing at the top level
        with Manager() as manager:
            shared_best_solution = manager.list(best_solution)
            shared_len_best_solution = manager.Value('i', len_best_solution)
            
            with Pool() as pool:
                args = [
                    (state, game, moves + [move], shared_best_solution, shared_len_best_solution.value, depth + 1)
                    for move in unique_moves
                ]
                results = pool.map(solve_parallel, args)

            # Consolidate results
            for result in results:
                if len(result[0]) < shared_len_best_solution.value:
                    shared_best_solution[:] = result[0]
                    shared_len_best_solution.value = len(result[0])
            
            return list(shared_best_solution), shared_len_best_solution.value

    # Without multiprocessing for deeper recursive calls
    for move in unique_moves:
        moves.append(move)
        new_state, done = game.step(move)

        if done:
            if len(moves) < len_best_solution:
                best_solution = moves[:]
                len_best_solution = len(best_solution)
        else:
            best_solution, len_best_solution = solve(new_state, game, moves, best_solution, len_best_solution, depth + 1)

        # Reset for the next search
        game.reset(state)
        moves.pop()

    return best_solution, len_best_solution


    #print(moves)
    if depth < 11:
        if type(states) == bool:
            #print(len(moves), len(current_solution))
            if states == True:
        
                #print(moves)
                if len(current_solution) > len(moves):
                    
                    current_solution = moves[:]

        else:
            for state in states:
                moves.append(state[0])
                current_solution = unwrap(state[1], moves, depth+1, current_solution)
    if moves != []:
        moves.pop(-1)
    return current_solution    

def main():
    

    #state = game.get_state()
    """
    original_state =  np.array([
    [2, 1, 2, 3, 3, 3, 2],
    [1, 2, 2, 4, 4, 2, 3],
    [1, 4, 2, 4, 1, 3, 1],
    [4, 4, 1, 4, 1, 2, 2],
    [3, 1, 4, 2, 3, 1, 3],
    [1, 3, 1, 3, 1, 1, 1],
    [1, 3, 4, 3, 2, 1, 3],
    [3, 3, 2, 2, 2, 2, 3],
    [3, 3, 4, 2, 3, 1, 1]
    ])
    """
    
    original_state =  np.array([
    [3, 1, 2, 3, 1, 2, 2],
    [4, 1, 3, 1, 1, 2, 3],
    [3, 1, 3, 2, 4, 4, 3],
    [1, 4, 1, 1, 1, 4, 1],
    [2, 1, 2, 2, 2, 4, 4],
    [1, 4, 1, 1, 2, 1, 4],
    [4, 1, 1, 2, 2, 3, 4],
    [3, 1, 1, 3, 2, 2, 1],
    [4, 1, 2, 4, 3, 3, 4]
    ])
    """
    original_state =  np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 2, 2, 3],
    [0, 0, 4, 2, 2, 1, 3],
    [0, 0, 4, 1, 2, 3, 1],
    [0, 0, 2, 3, 2, 2, 1],
    [0, 0, 1, 4, 3, 3, 4]
    ])
    """
    game = FormerGame(original_state, grid_size=(7, 9), num_symbols=4)
    print("Initial game state:")
    print(original_state)

    #action = (int(input()), int(input()))
    #new_state, done = game.step(action)

    #games = [([move, move,...], grid)]
    #print(game.get_valid_actions())

    shortest_game = 17
    visited_states = set()

    
    start_new = time.time()
    solution, len_solution = solve(original_state, game,[], [], shortest_game, 0)
    time_new = time.time() - start_new
    
    print(f"\n Shortest path to clear the board:{solution}")

    print(time_new)
    #game.reset(original_state)



    for move in solution:
        print(game.grid[move[0],move[1]])

        new_state, done= game.step(move)
        

        print("\n",move,"\n", new_state,done, "\n")

    """
    start_states = time.time()
    states = get_states(original_state, game, shortest_game, 0)
    time_states = time.time() - start_states 

    #for state in states:
    #    print("\n",state)

    
    start_solution = time.time()
    current_solution = unwrap(states, [], 0, [0]*15)
    time_solution = time.time() - start_solution

    print(f"\n Shortest path to clear the board:{current_solution}")

    print(time_states, time_solution)
    #game.reset(original_state)


 
    for move in current_solution:
        print(game.grid[move[0],move[1]])

        new_state, done= game.step(move)
        

        print("\n",move,"\n", new_state,done, "\n")
    """
    
 
if __name__ == "__main__":
   
    main()