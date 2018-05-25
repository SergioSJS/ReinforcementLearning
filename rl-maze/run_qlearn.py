import random

from maze_env import Maze
from rl_brain import QLearningTable
#reload(Maze)

MAX_EPISODES = 100
actionSpace = []

def update():
    for episode in range(MAX_EPISODES):
        # Initial state
        state = env.reset()
        # Initial reward
        totReward = 0
        while True:
            # Fresh maze
            env.render()
            # RL choose action based on state
            action = RL.choose_action(str(state))
            # action = random.choice(actionSpace)
            # RL take action and get next state and reward
            state_, reward, done = env.step(action)

            # Rl learn from this transition
            totReward += reward
            RL.learn(str(state), action, totReward, str(state_))
            

            # change old state
            state = state_
            # break when done
            if done:
                print episode, totReward
                break
    # end of game
    print('game over')
    env.destroy()

if __name__ == '__main__':    
    env = Maze()
    actionSpace = env.actionSpace
    RL = QLearningTable(actionsList=actionSpace, learningRate=1, rewardDecay=0.95, eGreedy=0.95)
    env.after(100, update)
    env.mainloop()