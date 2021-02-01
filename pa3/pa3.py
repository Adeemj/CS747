import argparse
import numpy as np
import matplotlib.pyplot as plt

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
UP_RIGHT = 4
DOWN_RIGHT = 5
DOWN_LEFT = 6
UP_LEFT = 7
dict_direction2delta = {UP:[-1,0], RIGHT:[0,1], DOWN:[1,0], LEFT:[0,-1],
                        UP_RIGHT:[-1,1], DOWN_RIGHT:[1,1], DOWN_LEFT:[1,-1], UP_LEFT:[-1,-1]}

class WindyGrid():
    """
    This is a simple implementation of the Windy grid
    reinforcement learning task.
    Adapted from Example 6.4 from Reinforcement Learning:
    An Introduction by Sutton and Barto:
   
    With inspiration from:
    OpenAI Gym
    The board is a 7x10 matrix, with (using Numpy matrix indexing):
        [3, 0] as the start 
        [3, 7] as the goal 
        [0, 0, 0, 1, 1, 1, 2, 2, 1, 0] as the wind directions for cols
    Each time step incurs -1 reward. An episode terminates when the agent reaches the goal.
    """

    def __init__(self, king=False, stochastic=False):
        self.king = king
        self.discount = 1
        rows = 7
        cols = 10
        self.shape = (rows, cols)
        self.start_state = (3, 0)
        self.start_state_index = np.ravel_multi_index(self.start_state, self.shape)
        self.terminal_state = (3, 7)
        self.terminal_state_index = np.ravel_multi_index(self.terminal_state, self.shape)
        nS = np.prod(self.shape)
        if self.king:
            nA = 8
        else:
            nA = 4
        #wind
        self.wind_dir = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            col = position[1]
            wind_delta = self.wind_dir[col]
            P[s] = {a: [] for a in range(nA)}
            for a in range(nA):
                delta = dict_direction2delta[a]
                if stochastic and wind_delta>0:
                    for error in [-1, 0, 1]:
                        P[s][a].append(self._calculate_transition_prob(position, [delta[0]-wind_delta + error, delta[1]], 1.0/3))
                else:
                    P[s][a].append(self._calculate_transition_prob(position, [delta[0]-wind_delta, delta[1]], 1.0))

        self.P = P
        self.lastaction = None # for rendering
        self.nS = nS
        self.nA = nA

        # self.seed(seed)
    
    def _limit_coordinates(self, coord):
        
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, p):
        
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)

        is_done = tuple(new_position) == self.terminal_state
        return (p, new_state, -1, is_done)
    
    def step(self, a):
        transitions = self.P[self.s][a]

        i = np.random.choice(list(range(len(transitions))), p = [t[0] for t in transitions])
        p, s, r, d= transitions[i]
        #for deterministic
        # p, s, r, d= transitions[0]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob" : p})

class Trainer:

    def __init__(self, env):
        self.env = env
        self.time_step = 0
    
    def _epsilon_greedy(self, state, epsilon):
        #with epsilon probability we explore
        if bool(np.random.binomial(1, 1-epsilon)):
            #take greedy choice
            action = np.argmax(self.Q[state, :])
        else:
            #take random choice
            action = np.random.choice(self.env.nA)
        return int(action)

    def _update(self, trans_tup, algorithm, lr, epsilon):
        prev_state, action, reward, next_state, next_action = trans_tup
        gamma = self.env.discount
        if algorithm == 'Q-learning':
            target = reward + gamma*np.max(self.Q[next_state, :])
        elif algorithm == 'sarsa':
            target = reward + gamma*self.Q[next_state, next_action]
        elif algorithm == 'expected-sarsa':
            exp_Q = ( np.mean(epsilon*self.Q[next_state, :])
                    + (1-epsilon)*np.max(self.Q[next_state, :]) )
            target = reward + gamma*exp_Q
        self.Q[prev_state, action] += lr* (target - self.Q[prev_state, action])
    
    def _plot_path(self, path, total_reward):
        path = np.array(path)
        print(path[0,0], path[0, 1])
        plt.ylim(self.env.shape[0], 0)
        plt.xlim(0, self.env.shape[1])
        plt.plot(path[:, 1], path[: ,0])
        plt.show()
        print('Total reward = ', total_reward)

    def train(self, episodes, epsilon, lr, algorithm, seed, plot_path):
        #init
        np.random.seed(seed)
        self.Q = np.zeros((self.env.nS, self.env.nA))
        time_steps_episode = np.zeros(episodes)

        for episode in range(episodes):
            time_step = 0
            total_reward = 0
            path = []
            self.env.s = self.env.start_state_index

            action = self._epsilon_greedy(self.env.s, epsilon)
            while(True):
                prev_state = self.env.s
                prev_coord = np.unravel_index(prev_state, self.env.shape)
                next_state, reward, done, p =  self.env.step(action)
                
                next_action = self._epsilon_greedy(self.env.s, epsilon)
                trans_tup = (prev_state, action, reward, next_state, next_action)
                self._update(trans_tup, algorithm, lr, epsilon)
                action = next_action

                #for plotting and measuring performance
                path.append(prev_coord)
                time_step+=1
                total_reward+=reward

                if done:
                    path.append(self.env.terminal_state)
                    time_steps_episode[episode] = time_step
                    if plot_path and episode%(episodes//10)==0:
                        self._plot_path(path, total_reward)
                    break
                
        return time_steps_episode
    
    def plot_optimal_strategy(self ):
        self.train(episodes=1000, epsilon=0.1, lr=0.5,
                   algorithm='sarsa', seed=0, plot_path=False)
        optimal_strategy_row = np.zeros(self.env.shape)
        optimal_strategy_col = np.zeros(self.env.shape)
        for s in range(self.env.nS):
            position_row, position_col = np.unravel_index(s, self.env.shape)
            optimal_action = np.argmax(self.Q[s,:])
            delta = dict_direction2delta[optimal_action]
            optimal_strategy_row[position_row, position_col] = delta[0]
            optimal_strategy_col[position_row, position_col] = delta[1]
        y = np.arange(0, self.env.shape[0], 1)
        x = np.arange(0, self.env.shape[1], 1)

        plt.quiver(x, y, optimal_strategy_col, optimal_strategy_row)
        plt.show()
        

def calc_cumulative_data(time_steps_episode):
    episodes = len(time_steps_episode)
    total_time_steps = np.sum(time_steps_episode)
    cum_data = np.zeros(total_time_steps)
    cum_t = 0
    for episode in range(episodes):
        cum_data[cum_t:+cum_t+time_steps_episode[episode]] = episode
        cum_t += int(time_steps_episode[episode])
    return cum_data

def run_experiment(king=False, stochastic=False, episodes=175, epsilon=0.1,
                   lr=0.5, algorithm='sarsa', random_iterations=10):
    my_windy_grid = WindyGrid(king, stochastic)
    my_train = Trainer(my_windy_grid)
    mean_steps_episode = np.zeros(episodes)
    for seed in range(random_iterations):
        time_steps_episode = my_train.train(episodes=episodes, epsilon=0.1,
                        lr=0.5, algorithm=algorithm, seed=seed, plot_path=False)
        mean_steps_episode += time_steps_episode
    mean_steps_episode /= random_iterations
    mean_steps_episode = (np.round(mean_steps_episode)).astype(int)
    cum_data = calc_cumulative_data(mean_steps_episode)
    return cum_data

def main():
    parser = argparse.ArgumentParser(description='Planner')
    parser.add_argument('-t', '--task', type=int, required=True, help='Task number')
    
    args = parser.parse_args()

    task = args.task

    if task == 2:
        #task2
        plt.figure(figsize=(10,5))
        cum_data = run_experiment()
        plt.plot(cum_data, label='Basic')
        plt.title('Windy GridWorld, learning rate=0.5, epsilon=0.1')
        plt.xlabel('Time steps')
        plt.ylabel('Episodes')
        plt.legend()
        plt.savefig('basic')
        plt.show()

    elif task == 3:
        #task 3
        plt.figure(figsize=(10,5))
        cum_data = run_experiment(king=True)
        plt.plot(cum_data, label="King's")
        plt.title('Windy GridWorld, learning rate=0.5, epsilon=0.1')
        plt.xlabel('Time steps')
        plt.ylabel('Episodes')
        plt.legend()
        plt.savefig('kings.png')
        plt.show()
    

    elif task == 4:
        #task 4
        plt.figure(figsize=(10,5))
        cum_data = run_experiment(king=True, stochastic=True)
        plt.plot(cum_data, label="King's + Stochastic")
        plt.title('Windy GridWorld, learning rate=0.5, epsilon=0.1')
        plt.xlabel('Time steps')
        plt.ylabel('Episodes')
        plt.legend()
        plt.savefig('kings_and_stochastic.png')
        plt.show()

    elif task == 5:
        #task 5
        plt.figure(figsize=(10,5))
        for algorithm in ['sarsa', 'expected-sarsa', 'Q-learning']:
            cum_data = run_experiment(algorithm=algorithm)
            plt.plot(cum_data, label=algorithm)
        plt.title('Windy GridWorld, learning rate=0.5, epsilon=0.1')
        plt.xlabel('Time steps')
        plt.ylabel('Episodes')
        plt.legend()
        plt.savefig('compare_algos.png')
        plt.show()
    
    elif task == 6:
        plt.figure(figsize=(10,5))
        cum_data = run_experiment()
        plt.plot(cum_data, label='base')
        cum_data = run_experiment(king=True)
        plt.plot(cum_data, label='king')
        cum_data = run_experiment(king=True, stochastic=True)
        plt.plot(cum_data, label='king+stochastic')
        plt.legend()
        plt.savefig('compare_env.png')
        plt.show()

    
    else:
        print('Task is an integer in [2,3,4,5,6]')
    

#driver code
if __name__ == "__main__":

    main()



