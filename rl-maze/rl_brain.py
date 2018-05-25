import numpy as np
import pandas as pd

class QLearningTable():
    def  __init__(self, actionsList, learningRate=1, rewardDecay=0.95, eGreedy=0.95):
        self.actionsList = actionsList
        self.lr = learningRate
        self.gamma = rewardDecay
        self.epsilon = eGreedy
        self.qTable = pd.DataFrame(columns=self.actionsList, dtype=np.float64)
        print self.qTable

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # Action select
        # If the greedy
        if np.random.uniform() < self.epsilon:
            # choose the best action
            stateAction = self.qTable.loc[observation, :]
            stateAction = stateAction.reindex(np.random.permutation(stateAction.index))
            action = stateAction.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actionsList)

        return action
        

    def learn(self, observation, action, reward, observation_):
        self.check_state_exist(observation_)
        qPredict = self.qTable.loc[observation, action]
        if observation_ != 'finished':
            qTarget = reward + self.gamma*self.qTable.loc[observation_, :].max()
        else:
            qTarget = reward
        # Update learning
        self.qTable.loc[observation, action] += (self.lr * (qTarget - qPredict))

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