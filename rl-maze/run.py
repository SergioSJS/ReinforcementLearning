import random

from maze_env import Maze
from rl_brain import QLearningTable
#reload(Maze)

MAX_EPISODES = 100
actionSpace = []

def update():
    for episode in range(MAX_EPISODES):
        # Initial observation
        observation = env.reset()
        # Initial reward
        totReward = 0
        while True:
            # Fresh maze
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            # action = random.choice(actionSpace)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # Rl learn from this transition
            totReward += reward
            RL.learn(str(observation), action, totReward, str(observation_))
            

            # change old observation
            observation = observation_
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
    RL = QLearningTable(actionsList=actionSpace)
    env.after(100, update)
    env.mainloop()