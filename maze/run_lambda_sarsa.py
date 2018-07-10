import random

from maze_env import Maze
from rl_brain import SarsaLambdaTable
#reload(Maze)

MAX_EPISODES = 100
actionSpace = []

def update():
    for episode in range(MAX_EPISODES):
        # Initial state
        state = env.reset()
        # Initial reward
        totReward = 0
        # RL choose action based on observation of state
        action = RL.choose_action(str(state))
        # initial all zero eligibility trace
        RL.eligibilityTrace *= 0
        while True:
            # Fresh maze
            env.render()
            
            # RL take action and get next state and reward
            state_, reward, done = env.step(action)

            # RL choose action based on next state
            action_ = RL.choose_action(str(state_))
            # action = random.choice(actionSpace)

            # Rl learn from this transition
            totReward += reward
            RL.learn(str(state), action, totReward, str(state_), action_)
            

            # change old state and action
            state = state_
            action = action_
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
    RL = SarsaLambdaTable(actionsList=actionSpace, learningRate=0.1, rewardDecay=0.95, eGreedy=0.95)
    env.after(100, update)
    env.mainloop()