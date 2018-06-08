
'''
Inspired:
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/

File wit the classical methods
'''
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, merge, Input
from keras.optimizers import Adam
from keras import backend as K

np.random.seed(7)
tf.set_random_seed(7)

# Tradicional Reinforcement Learning
class RL(object):
    def  __init__(self, actionsList, learningRate=1, rewardDecay=0.95, eGreedy=0.95):
        self.actionsList = actionsList
        self.lr = learningRate
        self.gamma = rewardDecay
        self.epsilon = eGreedy
        self.qTable = pd.DataFrame(columns=self.actionsList, dtype=np.float64)
    
    # Return action 
    # Exploration: random action 
    # Exploitation: greedy action
    def choose_action(self, state):
        self.check_state_exist(state)
        # Action select
        # If the greedy
        if np.random.uniform() < self.epsilon:
            # choose the best action
            stateAction = self.qTable.loc[state, :]
            stateAction = stateAction.reindex(np.random.permutation(stateAction.index))
            action = stateAction.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actionsList)

        return action

    # Overwrite for each method
    def learn(self, *args):
        pass
    
    # Check is exist, add in qTable if there is not
    def check_state_exist(self, state):
        if state not in self.qTable.index:
            # append new state on qTable
            self.qTable = self.qTable.append(
                pd.Series(
                    [0]*len(self.actionsList),
                    index=self.qTable.columns,
                    name=state,
                )
            )

# Off-policy - Q-learning
class QLearningTable(RL):
    def __init__(self, actionsList, learningRate=1, rewardDecay=0.95, eGreedy=0.95):
        super(QLearningTable, self).__init__(actionsList, learningRate, rewardDecay, eGreedy)

    def learn(self, state, action, reward, state_):
        self.check_state_exist(state_)
        qPredict = self.qTable.loc[state, action]
        if state_ != 'finished':
            qTarget = reward + self.gamma*self.qTable.loc[state_, :].max()
        else:
            qTarget = reward
        # Update learning
        self.qTable.loc[state, action] += (self.lr * (qTarget - qPredict))

# On-policy - Sarsa (State Action Reward State Action)
class SarsaTable(RL):
    def __init__(self, actionsList, learningRate=1, rewardDecay=0.95, eGreedy=0.95):
        super(SarsaTable, self).__init__(actionsList, learningRate, rewardDecay, eGreedy)

    def learn(self, state, action, reward, state_, action_):
        self.check_state_exist(state_)
        qPredict = self.qTable.loc[state, action]
        if state_ != 'finished':
            qTarget = reward + self.gamma*self.qTable.loc[state_, action_]
        else:
            qTarget = reward
        
        self.qTable.loc[state,action] += self.lr*(qTarget-qPredict)

# On-policy - Sarsa (State Action Reward State Action)
class SarsaLambdaTable(RL):

    def __init__(self, actionsList, learningRate=1, rewardDecay=0.95, eGreedy=0.95, traceDecay=0.9):
        super(SarsaLambdaTable, self).__init__(actionsList, learningRate, rewardDecay, eGreedy)
        self.lambda_ = traceDecay
        self.eligibilityTrace = self.qTable.copy()

    def check_state_exist(self, state):
        if state not in self.qTable.index:
            # append new state on qTable
            item = pd.Series(
                    [0]*len(self.actionsList),
                    index=self.qTable.columns,
                    name=state,
                )
            self.qTable = self.qTable.append(item)
            # Update elegibility trace
            self.eligibilityTrace = self.eligibilityTrace.append(item)

    def learn(self, state, action, reward, state_, action_):
        self.check_state_exist(state_)
        qPredict = self.qTable.loc[state, action]
        if state_ != 'finished':
            qTarget = reward + self.gamma*self.qTable.loc[state_, action_]
        else:
            qTarget = reward

        error = qTarget-qPredict
        
        # Methode 1:
        self.eligibilityTrace.loc[state, action] += 1

        # Methode 2:
        '''self.eligibilityTrace.loc[state,:] *= 0
        self.eligibilityTrace.loc[state, action] = 1'''

        # Update Q table
        self.qTable += self.lr * error * self.eligibilityTrace
        # Update elegibility trace
        self.eligibilityTrace *= self.gamma*self.lambda_ 