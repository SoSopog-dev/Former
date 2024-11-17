import numpy as np

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
        valid_actions = []
        rows, cols = self.grid.shape
        for row in range(rows):
            for col in range(cols):
                symbol = self.grid[row, col]
                if symbol != 0:
                    valid_actions.append((row, col))
        return valid_actions  

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
    moves = game.get_valid_actions()

    for m in moves:
        if m in moves:
            alike_moves = game.find_connected_blocks(m[0], m[1], game.grid[m[0], m[1]])
            alike_moves.remove(m)

            moves = [move for move in moves if move not in alike_moves]

    return moves

def get_states(current_state, game, shortest_path, depth, visited_states):
    if shortest_path < depth:
        return current_state
    #print(f"current_state nr.1 :{current_state}")
    states = []
    for move in find_unique_moves(game):
        #print(f"\n Unique move given this position: {current_state}, {move}\n")
        temp = game.step(move)
        new_state, done = temp[0], temp[1]
        if new_state in visited_states:
            return False
        else:
            visited_states.add(new_state)
        if done:
            shortest_path = depth
            game.reset(current_state)    
            states.append((move, True))
            return states
        states.append((move, get_states(new_state, game, shortest_path, depth + 1)))
        game.reset(current_state)        
        ##print(f"current_state nr.2 :{current_state}")
    return states

def unwrap(states, moves, depth, current_solution):
    #print(moves)
    if depth >= 11:
        moves.pop(-1)
    else:
        if type(states) == bool:
            #print(len(moves), len(current_solution))
            
            if states == True:
                if len(current_solution) > len(moves):
                    print(moves)
                    current_solution = moves[:]
                moves.pop(-1)
                #print(f"\nWe have now poped an element from moves:{moves}\n")
            else:
                moves.pop(-1)
        else:
            for state in states:
                moves.append(state[0])
                current_solution = unwrap(state[1], moves, depth+1, current_solution)
                if not moves == []:
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
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 2, 3, 3, 2, 1, 2],
    [3, 2, 1, 1, 1, 2, 3],
    [3, 3, 2, 1, 3, 3, 3]
    ])
    
    game = FormerGame(original_state, grid_size=(7, 9), num_symbols=4)
    print("Initial game state:")
    print(original_state)

    #action = (int(input()), int(input()))
    #new_state, done = game.step(action)

    #games = [([move, move,...], grid)]

    shortest_game = 13
    visited_states = {}
    states = get_states(original_state, game, shortest_game, 0, visited_states)
    #print(states)


    #print(states)

    
    
    current_solution = unwrap(states, [], 0, [0]*13)

    print(current_solution)
    print("\n Shortest path to clear the board:")

    #game.reset(original_state)
    """
    for move in current_solution:
        print(game.grid[move[0],move[1]])
        new_state= game.step(move)[0]

        print("\n",move,"\n", new_state, "\n")
    """
 
if __name__ == "__main__":
   
    main()