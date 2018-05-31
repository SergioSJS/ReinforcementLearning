import random

from maze_env import Maze
from rl_brain import DeepQNetworkKeras
#reload(Maze)

MAX_EPISODES = 10000
actionSpace = []

def update():
    step = 0
    for episode in range(MAX_EPISODES):
        # Initial state
        state = env.reset()
        # Initial reward
        totReward = 0
        
        while True:
            # Fresh maze
            env.render()
            
            # RL choose action based on observation of state
            action = RL.choose_action(state)

            # RL take action and get next state and reward
            state_, reward, done = env.step(actionSpace[action])

            # Rl learn from this transition
            totReward += reward

            # Storing information
            RL.storeTransition(state, action, totReward, state_)
           
            if(step>200) and (step % 5 == 0):
                RL.learn() 

            # change old state
            state = state_

            # break when done
            if done:
                print episode, totReward
                break
            step += 1
    # end of game
    print('game over')
    env.destroy()

if __name__ == '__main__':    
    env = Maze()
    actionSpace = env.actionSpace
    RL = DeepQNetworkKeras(
        actionsList=actionSpace, nFeatures=env.nFeatures,
        learningRate=0.01, 
        rewardDecay=0.90, 
        eGreedy=0.90,
        replaceTargetIter=200,
        memorySize=2000,
        batchSize=100)
    env.after(100, update)
    env.mainloop()