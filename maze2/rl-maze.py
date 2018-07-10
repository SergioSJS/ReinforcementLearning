
# coding: utf-8
"""
@author: SergioSJS - Sérgio José de Sousa
@email: sergio7sjs@gmail.com
Artificial Intelligence Discipline Work
UFMG - Teacher: Luiz Chaimowicz
"""
import numpy as np
import time
import copy
import random
import pandas as pd
import sys
from IPython.display import clear_output

class Maze():
    '''Maze environment. 
    Use a text file to generate a maze, having a four elements, wall, enemy, coin and free.
    This environment have four actions, up, down, left rignt.
    '''
    def __init__(self, maze_dir, step_cost = -1, agent_pos_ini=[]):
        '''Initialize maze
        
        Keyword arguments:
        maze_dir -- maze text file
        step_cost -- step cost to free way (default -1)
        agent_pos_ini -- defines a specific starting point (default [])
        '''
        # action list
        self.action_space = ['u', 'd', 'l', 'r']
        self.maze_dir = maze_dir
        self.build_maze()
        
        # elements of the maze
        self.wall = '#'
        self.enemy = '&'
        self.coin = '0'
        self.free = '-'
                
        if agent_pos_ini == []:
            self.agent_pos_ini = self.random_position()
        else:
            self.agent_pos_ini = agent_pos_ini
            
        self.agent_pos = copy.copy(self.agent_pos_ini)
        
        self.step_cost = step_cost
        
    def build_maze(self):
        '''Read the file and build a maze as an array
        '''
        maze_file = open(self.maze_dir, "r").read().splitlines()
        line = 1
        mMap = None
        for l in maze_file:
            if line == 1:
                self.mapH = int(l.split(" ")[0])
                self.mapW = int(l.split(" ")[1])        
                mMap = np.empty((0, self.mapW))
            else:
                mLine = np.array(list(l))
                mMap = np.append(mMap, [mLine], axis=0)
            line += 1
        self.map = mMap
    
    def render(self, clear=False):
        '''Render maze
        
        Keyword arguments:
        clear -- If clear console of jupyter notebook
        '''
        if clear:
            clear_output()
        map_temp = self.map.copy()

        map_temp[self.agent_pos[0],self.agent_pos[1]] = 'X'
        for l in map_temp:
            print(''.join(l))
            
    def random_position(self):
        '''Generate a valid random position
        return -- [x,y] position
        '''
        while True:
            rW = random.randrange(self.mapW)
            rH = random.randrange(self.mapH)

            if self.map[rH][rW] == self.free:
                return [rH, rW]
                break
    
    def reset(self):
        '''Reset environment
        return -- first state
        '''
        state = self.random_position()
        self.agent_pos_ini = state
        self.agent_pos = copy.copy(state)
        return copy.copy(state)
    
    def step(self, action):
        '''Performs the action
        
        Keyword arguments:
        action -- valid action ['u', 'd', 'l', 'r']
        
        return -- aget_pos -- after action, return new state
                  reward -- reward of state
                  done -- if is a terminal state
        '''
        pos = copy.copy(self.agent_pos)
        reward = 0
        done = False
        
        # calculate the destiny coordinate
        if (action == 'u'):
            pos[0] = pos[0] - 1
        elif (action == 'd'):
            pos[0] = pos[0] + 1            
        elif (action == 'l'):
            pos[1] = pos[1] - 1            
        elif (action == 'r'):
            pos[1] = pos[1] + 1
        
        # verify if is valid and calculate the reward and if done
        if(pos[0] < self.mapH or pos[0] >= 0 or 
           pos[1] < self.mapW or pos[1] >= 0):
            if self.map[tuple(pos)] == self.wall:
                reward = -1
            elif self.map[tuple(pos)] == self.coin:
                reward = 10
                done = True              
            elif self.map[tuple(pos)] == self.enemy:
                reward = -10
                done = True                        
            else:
                reward = -1
                self.agent_pos = pos
                
        return self.agent_pos, reward, done
    
class ReinforcementLearning(object):
    '''Base class of Reinforcement Learning methods
    '''
    def __init__(self, action_space, learning_rate, reward_decay, greedy):
        '''Initialize Reinforcement Learning object
        
        Keyword arguments:
        action_space -- list of possible actions
        learning_rate -- alpha
        reward_decay -- episilon
        greedy -- greedy probability
        '''
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = greedy
        #Q Table
        self.q_table = pd.DataFrame(columns=self.action_space, dtype=np.float64)
    
    def choose_action(self, state):
        '''According of state, select the action
        
        Keyword arguments:
        state -- current state
        
        return -- action -- best action to take
        '''
        self.check_state(state)
        # Action select
        # if random
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.action_space)
        # if greedy
        else:
            state_actions = self.q_table.loc[state, :]
            state_actions = state_actions.reindex(np.random.permutation(state_actions.index))
            action = state_actions.idxmax()
        
        return action
        
    def check_state(self, state):
        '''Verifies if the state exists and add if not.
        
        Keyword arguments:
        state -- state to check
        '''
        if state not in self.q_table.index:
            # Append new state on q_table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.action_space),
                    index=self.q_table.columns,
                    name=state,
                )
            )
            
    def learn(self, *args):
        '''Method to learn
        '''
        pass
    
class QLearningTable(ReinforcementLearning):
    '''Off-policy - Q-learning
    '''
    def __init__(self, action_space, learning_rate, reward_decay, greedy):
        '''Initialize Q Learning object
        
        Keyword arguments:
        action_space -- list of possible actions
        learning_rate -- alpha
        reward_decay -- episilon
        greedy -- greedy probability
        '''
        super(QLearningTable, self).__init__(action_space, learning_rate, reward_decay, greedy)
    
    def learn(self, old_state, action, reward, state, done):
        '''Learning method
        
        Keyword arguments:
        old_state -- old state before taking action
        action -- action performed
        reward -- reward received
        state -- state after action
        done -- if terminal state
        '''
        self.check_state(state)
        q_predict = self.q_table.loc[old_state, action]
        
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * self.q_table.loc[state,:].max()
            
        # Update q value
        self.q_table.loc[old_state, action] += (self.lr * (q_target - q_predict))
        
def print_pi(maze, q_table, text=True, verbose=False):
    '''Print maze with the max q value
    
    Keyword arguments:
    maze -- Maze Environment
    q_table -- Reinforcement Learning q_table
    text -- save text file pi.txt (default True)
    verbose -- if print on console (default False)
    '''
    w = 0
    h = 0
    map_tmp = copy.copy(maze.map)
    for h in range(maze.mapH):    
        for w in range(maze.mapW):
            state = [h,w]
            if str(state) in q_table.index:
                state_actions = q_table.loc[str(state), :]
                action = state_actions.idxmax()
                if action == 'u':
                    map_tmp[h][w] = '^'
                elif action == 'd':
                    map_tmp[h][w] = 'v'
                elif action == 'l':
                    map_tmp[h][w] = '<'
                elif action == 'r':
                    map_tmp[h][w] = '>'
    if text:
        text_file = open("pi.txt", "w")

    for l in map_tmp:
        if verbose:
            print(''.join(l))
        if text:
            text_file.write(''.join(l)+'\n')
    if text:
        text_file.close()
                
def print_q(maze, q_table, text=True, verbose=False):
    '''Print q values of states
    
    Keyword arguments:
    maze -- Maze Environment
    q_table -- Reinforcement Learning q_table
    text -- save text file pi.txt (default True)
    verbose -- if print on console (default False)
    '''
    w = 0
    h = 0
    map_tmp = copy.copy(maze.map)
    
    if text:
        text_file = open("q.txt", "w")
        
    for h in range(maze.mapH):    
        for w in range(maze.mapW):
            state = [h,w]
            if str(state) in q_table.index:
                state_actions = q_table.loc[str(state), :]
                
                for action, ac in [['acima', 'u'],['abaixo', 'd'],['esquerda', 'l'],['direita', 'r']]:                
                    value = state_actions.get(ac)

                    if verbose:
                        print(str(h)+','+str(w)+','+action+','+str(value))
                    if text:
                        text_file.write(str(h)+','+str(w)+','+action+','+str(value)+'\n')
                    
                
    if text:
        text_file.close()

def run(map_dir, alpha, gamma, max_steps, 
        epsilon_steps = 1,
        epsilon_initial = 0.05,
        epsilon_final = 0.05,
        just_step_cost=False):
    '''
    Keyword arguments:
    map_dir -- Map directory
    alpha -- Learning Rate
    gamma -- Reward decay
    max_steps -- Number max of steps
    epsilon_steps -- Number of steps over which the initial value of epsilon is linearly annealed to its final value
    epsilon_initial -- Initial value of epsilon in epsilon-greedy
    epsilon_final -- Final value of epsilon in epsilon-greedy    
    '''
    # Initialize environment object
    maze = Maze(map_dir)

    # Just step cost
    if just_step_cost:
        maze.step_cost = -1./(maze.mapH*maze.mapW)

    # Initialize Q learning object
    QL = QLearningTable(
        action_space = maze.action_space,
        learning_rate = alpha,
        reward_decay = gamma,
        greedy = 0.05
    )
    
    epsilon = epsilon_initial
    epsilon_step = (epsilon_initial - epsilon_final) / epsilon_steps

    # Number max of episodes
    MAX_EPISODES = 10000
    # Step counter
    step = 0
    # Episode loop
    for episode in range(MAX_EPISODES):
        # Initial state
        old_state = maze.reset()
        tot_reward = 0
        while True:
            # Define epsilon greedy
            QL.epsilon = epsilon
            # RL choose action based on state
            action = QL.choose_action(str(old_state))
            # RL apply the action in the environment
            state, reward, done = maze.step(action)
            # RL learn
            tot_reward += reward
            QL.learn(str(old_state), action, reward, str(state), done)
            # change old state
            old_state = state

            step += 1
            
            if epsilon > epsilon_final:
                epsilon -= epsilon_step
            
            # break when done
            if done:
                print('step:'+str(step)+' ep.:'+str(episode) + ' rew.:'+ str(tot_reward)+'           \r'),
                break
            if step >= max_steps:
                break
        if step >= max_steps:
            break
    print('game over                           ')
    
    return maze, QL


#Take the args and initializate variables
try:
    if len(sys.argv) >= 5:
        vMap = sys.argv[1]
        alpha = float(sys.argv[2])
        epsilon = float(sys.argv[3])
        max_steps = int(sys.argv[4])
    else:
        sys.exit("Error, args is missing!!")

except Exception as e:
    sys.exit("Error on args")

'''map_dir - Map directory
   alpha - Learning Rate
   epsilon -Reward decay
   max_steps - Number max of steps'''

maze, QL = run(vMap, alpha, epsilon, max_steps)


print_pi(maze, QL.q_table)
print_q(maze, QL.q_table)

