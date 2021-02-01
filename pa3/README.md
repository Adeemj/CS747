# Windy Grid World

This is a simple implementation of the Windy grid reinforcement learning task.
Adapted from Example 6.4 from Reinforcement Learning:
An Introduction by Sutton and Barto
Two alterations of the above environment have been implemented:
1)King's moves
2)Stochastic wind

Three learning algorithmms have been implemented:
1)sarsa
2)expected-sarsa
3)q-learning

## Installation

The only required libraries are numpy and matplotlib
```bash
pip install numpy
pip install matplotlib
```
## Usage
The environment and the trainer have been implemented as separate classes to allow independent usage

```python
#init WindyGrid with two args: 1)king: bool, allow king's moves
#2)stochastic: bool, stochastic winds
#following line creates an instance of WindyGrid with no kings moves but stochastic winds
my_windy_grid = WindyGrid(king=False, stochastic=True)

#init Trainer with one arg: env
my_trainer = Trainer(my_windy_grid)

#call trainer method with args: 1)episodes, 2)epsilon, 3)lr (learning rate)
#4)randomseed, 5)plot path (will plot path of agent 10 times overall)
#returns vector of length episodes with v[i] = number of steps in episode i
time_steps_episode = my_trainer.train(episodes=episodes, epsilon=0.1,
                        lr=0.5, algorithm=algorithm, seed=seed, plot_path=False)

#following are help functions for generating required plots

# following func gives the cumulative data req for the plots
cum_data = calc_cumulative_data(time_steps_episodes)

# following function generates data for 10 random iterations and returns avg cum data for base case. For info on other args please refer to code
cum_data = run_experiment()

#following is the code for plotting the results of the experiment
plt.figure(figsize=(10,5))
plt.plot(cum_data, label='Basic')
plt.title('Windy GridWorld, learning rate=0.5, epsilon=0.1')
plt.xlabel('Time steps')
plt.ylabel('Episodes')
plt.legend()
plt.savefig('basic')
plt.show()

```
## Command line script
TO run the program on your machine, navigate to this folder in bash/cmd and run:
```bash

python pa3.py --task i # i is the task number in [2,3,4,5]

```