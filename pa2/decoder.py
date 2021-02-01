import argparse
import numpy as np
from encoder import Encoder

#repeated codeblock------
from collections import OrderedDict
def tup2state(tup, rows, cols):
    i,j = tup
    return i*cols + j
def state2tup(state, rows, cols):
    return state//cols, state%cols
def action2nbors_dict(tup_state):
    i,j = tup_state
    action2nbor_state = {0:(i+1,j), 1:(i,j+1), 2:(i-1,j), 3:(i,j-1)}
    action2nbor_state_ord = OrderedDict(action2nbor_state.items())
    return action2nbor_state_ord
# -----------

def decoder(maze_matrix, policy):
    rows, cols = maze_matrix.shape
    path = []
    start_row, start_col = np.where(maze_matrix==2)
    start = tup2state((start_row[0], start_col[0]), rows, cols)

    end_row, end_col = np.where(maze_matrix==3)
    end = tup2state((end_row[0], end_col[0]), rows, cols)
    state = start

    while(True): #next state is deterministic
        action = policy[state]
        path.append(action)
        i, j = state2tup(state, rows, cols)
        next_state_tup = action2nbors_dict((i,j))[action]
        next_state = tup2state(next_state_tup, rows, cols)
        if next_state == end:
            break
        else:
            state = next_state
    req_map = {0:'S', 1:'E', 2:'N', 3:'W'}
    path = list(map(lambda x: req_map[x], path))
    path_str = ' '.join(path) 
    return path_str

def value_policy_from_txt(filepath):
    V = []
    pi = []
    with open(filepath) as fp:
        Lines = fp.readlines()
    for line in Lines:
        v_s, pi_s = line.strip().split()
        V.append(float(v_s))
        pi.append(int(pi_s))
    return np.array(V), np.array(pi)

def main():
    parser = argparse.ArgumentParser(description='Encode')
    parser.add_argument('-g', '--grid', type=str, required=True, help='Path of grid file')
    parser.add_argument('-pi', '--value_policy', type=str, required=True, help='Path of value and policy file')

    args = parser.parse_args()
    path_to_maze = args.grid
    my_encoder = Encoder(path_to_maze)
    maze_matrix = my_encoder.get_matrix()
    value_policy_file = args.value_policy
    _, policy = value_policy_from_txt(value_policy_file)

    # print(policy)
    print(decoder(maze_matrix, policy))

if __name__ == "__main__": 
    # calling the main function 
    main()

