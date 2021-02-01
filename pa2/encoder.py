import os
import argparse
from collections import OrderedDict
import numpy as np

class Encoder:
    def __init__(self, path_to_maze):
        self.path_to_maze = path_to_maze
        self._encode2matrix()
    
    def _encode2matrix(self):
        self.maze_matrix = []
        with open(self.path_to_maze) as fp:
            Lines = fp.readlines()
        for line in Lines:
            self.maze_matrix.append(list(map(int, line.strip().split() ) ))
        self.maze_matrix = np.array(self.maze_matrix, dtype='i1') #one byte 
        self.rows, self.cols = self.maze_matrix.shape
    
    def get_matrix(self):
        return self.maze_matrix

    def _tup2state(self, tup):
        i,j = tup
        return i*self.cols + j
    def _state2tup(self, state):
        return state//self.cols, state%self.cols

    def action2nbors_dict(self, tup_state):
        i,j = tup_state
        action2nbor_state = {0:(i+1,j), 1:(i,j+1), 2:(i-1,j), 3:(i,j-1)}
        action2nbor_state_ord = OrderedDict(action2nbor_state.items())
        return action2nbor_state_ord

    def encode(self):

        transitions = []
        for i in range(self.rows):
            for j in range(self.cols):
                this_state = self._tup2state((i,j))
                this_element = self.maze_matrix[i,j]
                if this_element == 1:
                    continue
                elif (this_element == 0) or (this_element == 2):
                    if this_element==2:
                        start = this_state
                    action2nbor_state_ord = self.action2nbors_dict((i,j))
                    for action, nbor_state_tup in action2nbor_state_ord.items():
                        nbor_element = self.maze_matrix[nbor_state_tup]
                        nbor_state = self._tup2state(nbor_state_tup)
                        if nbor_element == 1:
                            next_state = this_state
                            reward = 0
                        elif nbor_element == 0 or nbor_element == 2:
                            next_state = nbor_state
                            reward = 0
                        elif nbor_element == 3:
                            next_state = nbor_state
                            reward = 1e6
                        transitions.append({'state':this_state, 'action':action,
                                            'next_state':next_state, 
                                            'r': reward, 'p':1})

                elif this_element == 3:
                    end = this_state
                else:
                    print('INVALID ELEMENT')
                    exit()
            
        self.dict_mdp =  {'numStates': self.rows*self.cols,
                'numActions': 4, 'start': start, 'end': end,
                'transitions': transitions, 'mdptype':'episodic', 'discount':0.9}
        self.encode_write_to_txt()
        return self.dict_mdp

    def encode_write_to_txt(self):
        numStates = self.dict_mdp['numStates']
        numActions = self.dict_mdp['numActions']
        start = self.dict_mdp['start']
        end = self.dict_mdp['end']

        print(f'numStates {numStates}')
        print(f'numActions {numActions}')
        print(f'start {start}')
        print(f'end {end}')

        for transition in self.dict_mdp['transitions']:
            state = transition['state']
            action = transition['action']
            next_state = transition['next_state']
            r = transition['r']
            p = transition['p']
            print(f'transition {state} {action} {next_state} {r} {p}')

        mdptype = self.dict_mdp['mdptype']
        discount = self.dict_mdp['discount']

        print(f'mdptype {mdptype}')
        print(f'discount {discount}')

def main():
    parser = argparse.ArgumentParser(description='Encode')
    parser.add_argument('-g', '--grid', type=str, required=True, help='Path of grid file')
    
    args = parser.parse_args()
    path_to_maze = args.grid
    my_encoder = Encoder(path_to_maze)
    my_encoder.encode()

if __name__ == "__main__": 
    # calling the main function 
    main()

