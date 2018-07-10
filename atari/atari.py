import gym
import numpy as np
import random
import time
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Convolution2D, Flatten, Dense, Input, merge
from keras.optimizers import Adam
from keras import backend as K

from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque

IT = 2 # Number of iteration of simulation

FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
NUM_FRAMES = 4 # The number of most recent frames experienced by the agent that are giving as input to the Q network
NUM_ACTIONS = 18 # The number max of actions

EPSILON_STEPS = 200000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value #1000000
EPSILON_INITIAL = 1.0  # Initial value of epsilon in epsilon-greedy
EPSILON_FINAL = 0.1  # Final value of epsilon in epsilon-greedy

NUM_EPISODES = 10000 # Number max of episodes
NUM_EPISODES_AT_TEST = 10000 # Number of episodes at test

DQN_LEARNING_RATE = 0.000007 # The learning rate
DQN_UPDATE_TARGET = 10000 # The frequency with which the target network is updated
DQN_DECAY_RATE = 0.99 # Decay prediction
DQN_SAVE_EVERY = 10000 # Save DQN weights every number of steps

MEMORY_BATCH_SIZE = 32 # Number of training cases over which each (SGD) update is computed
MEMORY_SIZE = 300000 # Number of samples in memory
MEMORY_MIN_2_LEARN = 10000 # Number min of memories to learn 

env = gym.make(ENVIRONMENTS)
NUM_ACTIONS = env.action_space.n

env.reset()
for i, a in enumerate(env.unwrapped.get_action_meanings()):
    print (i, a)

class ReplayExperience:
    """Reinforcement Learning for Robots Using Neural Networks
    [http://www.dtic.mil/docs/citations/ADA261434]
    
    Buffer to stores the past moves, states and rewards
    """
    def __init__(self, experience_size):
        self.count = 0 # number of registers
        self.size = experience_size # max size
        self.memory = deque() # buffer
    
    def add(self, state_old, action, reward, state, done):
        experience = (state_old, action, reward, state, done)        
        if self.count < self.size:            
            self.count += 1
        else:
            self.memory.popleft()
            
        self.memory.append(experience)
    
    def sample(self, batch_size):
        sample = []
        if self.count < batch_size:
            sample = random.sample(list(self.memory), self.count)
        else:
            sample = random.sample(list(self.memory), batch_size)
            
        batch_state_old, batch_action, batch_reward, batch_state, batch_done = map(np.array, zip(*sample))
        return batch_state_old, batch_action, batch_reward, batch_state, batch_done     

class Atari2013():
    """ Playing Atari with Deep Reinforcement Learning
    link: https://arxiv.org/abs/1312.5602
    """
    def __init__(self):
        self.model = self.build_model()
        self.name = 'Atari2013'

    def build_model(self):
        model = Sequential()
        input_layer = Input(shape=(FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES))
        conv1 = Convolution2D(16, (8,8), strides=(4, 4), activation='relu')(input_layer)#data_format='channels_first'
        conv2 = Convolution2D(32, (4,4), strides=(2, 2), activation='relu')(conv1)
        flatten = Flatten()(conv2)
        fc1 = Dense(256, activation='relu')(flatten)
        fc2 = Dense(NUM_ACTIONS)(fc1)
        
        model = Model(inputs=[input_layer], outputs=[fc2])
        model.compile(loss='mse', optimizer=Adam(lr=DQN_LEARNING_RATE))
        
        return model

    def predict_action(self, state, epsilon):    
        if  np.random.random() <= epsilon:
            action = random.randrange(NUM_ACTIONS)
        else:
            predict = self.model.predict(state.reshape(1, FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES), batch_size = 1)
            action = np.argmax(predict)
        
        categorical_action = np.zeros(NUM_ACTIONS)
        categorical_action[action] = 1

        return action, categorical_action

    def learn(self, b_state_old, b_action, b_reward, b_state, b_done):
        """Train DQN
        """
        actions = self.model.predict_on_batch(b_state_old) 
        future_actions = self.model.predict_on_batch(b_state)
        
        for m in range(len(b_state_old)):
            if b_done[m]:
                b_action[m] = actions[m] + (b_action[m]*-actions[m]) + (b_action[m]* b_reward[m])
            else: 
                b_action[m] = actions[m] + (b_action[m]*-actions[m]) + (b_action[m]* DQN_DECAY_RATE * np.max(future_actions[m]))

        loss = self.model.train_on_batch(b_state_old, b_action)

    def save_network(self, path):
        # Saves model weights at specified path as h5 file
        self.model.save_weights(path)
        print("----Successfully saved network----")

    def load_network(self, path):
        # Load model weights at specified path as h5 file
        self.model.load_weights(path)
        print("----Successfully loaded network----")

class AtariDouble():
    """ Deep Reinforcement Learning with Double Q-learning
    link: https://arxiv.org/abs/1509.06461
    """
    def __init__(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.name = 'AtariDouble'

    def build_model(self):
        model = Sequential()
        input_layer = Input(shape=(FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES))
        conv1 = Convolution2D(32, (8,8), strides=(4, 4), activation='relu')(input_layer)#data_format='channels_first'
        conv2 = Convolution2D(64, (4,4), strides=(2, 2), activation='relu')(conv1)
        conv3 = Convolution2D(64, (3,3), strides=(1, 1), activation='relu')(conv2)
        flatten = Flatten()(conv3)
        fc1 = Dense(512, activation='relu')(flatten)
        fc2 = Dense(NUM_ACTIONS)(fc1)
        
        model = Model(inputs=[input_layer], outputs=[fc2])
        model.compile(loss='mse', optimizer=Adam(lr=DQN_LEARNING_RATE))
        
        return model

    def predict_action(self, state, epsilon):    
        if  np.random.random() <= epsilon:
            action = random.randrange(NUM_ACTIONS)
        else:
            predict = self.model.predict(state.reshape(1, FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES), batch_size = 1)
            action = np.argmax(predict)
        
        categorical_action = np.zeros(NUM_ACTIONS)
        categorical_action[action] = 1

        return action, categorical_action

    def learn(self, b_state_old, b_action, b_reward, b_state, b_done):
        """Train DQN
        """
        actions = self.model.predict_on_batch(b_state_old) 
        future_actions = self.target_model.predict_on_batch(b_state)
        
        for m in range(len(b_state_old)):
            if b_done[m]:
                b_action[m] = actions[m] + (b_action[m]*-actions[m]) + (b_action[m] * b_reward[m])
            else: 
                b_action[m] = actions[m] + (b_action[m]*-actions[m]) + (b_action[m] * DQN_DECAY_RATE * np.max(future_actions[m]))

        loss = self.model.train_on_batch(b_state_old, b_action)

    def target_model_update(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def save_network(self, path):
        # Saves model weights at specified path as h5 file
        self.model.save_weights(path)
        print("----Successfully saved network----")

    def load_network(self, path):
        # Load model weights at specified path as h5 file
        self.model.load_weights(path)
        print("----Successfully loaded network----")

class AtariDueling():
    """Dueling Network Architectures for Deep Reinforcement Learning
    link: https://arxiv.org/abs/1511.06581

    Missing implement:https://arxiv.org/abs/1511.05952
    """
    def __init__(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.name = 'AtariDueling'

    def build_model(self):
        model = Sequential()
        input_layer = Input(shape=(FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES))
        conv1 = Convolution2D(32, (8,8), strides=(4, 4), activation='relu')(input_layer)#data_format='channels_first'
        conv2 = Convolution2D(64, (4,4), strides=(2, 2), activation='relu')(conv1)
        conv3 = Convolution2D(64, (3,3), strides=(1, 1), activation='relu')(conv2)
        flatten = Flatten()(conv3)
        
        fc1 = Dense(512, activation='relu')(flatten)
        advantage = Dense(NUM_ACTIONS)(fc1)

        fc2 = Dense(512, activation='relu')(flatten)
        value = Dense(1)(fc2)

        policy = merge([advantage, value], 
            mode = lambda x: x[0]-K.mean(x[0])+x[1], output_shape = (NUM_ACTIONS,))
        
        model = Model(inputs=[input_layer], outputs=[policy])
        model.compile(loss='mse', optimizer=Adam(lr=DQN_LEARNING_RATE))
        
        return model

    def predict_action(self, state, epsilon):    
        if  np.random.random() <= epsilon:
            action = random.randrange(NUM_ACTIONS)
        else:
            predict = self.model.predict(state.reshape(1, FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES), batch_size = 1)
            action = np.argmax(predict)
        
        categorical_action = np.zeros(NUM_ACTIONS)
        categorical_action[action] = 1

        return action, categorical_action

    def learn(self, b_state_old, b_action, b_reward, b_state, b_done):
        """Train DQN
        """
        actions = self.model.predict_on_batch(b_state_old) 
        future_actions = self.target_model.predict_on_batch(b_state)
        
        for m in range(len(b_state_old)):
            if b_done[m]:
                b_action[m] = actions[m] + (b_action[m]*-actions[m]) + (b_action[m] * b_reward[m])
            else: 
                b_action[m] = actions[m] + (b_action[m]*-actions[m]) + (b_action[m] * DQN_DECAY_RATE * np.max(future_actions[m]))

        loss = self.model.train_on_batch(b_state_old, b_action)

    def target_model_update(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def save_network(self, path):
        # Saves model weights at specified path as h5 file
        self.model.save_weights(path)
        print("----Successfully saved network----")

    def load_network(self, path):
        # Load model weights at specified path as h5 file
        self.model.load_weights(path)
        self.target_model.load_weights(path)
        print("----Successfully loaded network----")



def load_model_save_weights(pathl, paths):
    model = load_model(pathl)
    print("----Successfully loaded network----")
    model.save_weights(paths)
    print("----Successfully saved network weights----")    
    
def train(tech):
    with open(tech.name+'_'+ENVIRONMENTS+'_episodes.csv', 'a') as file:
        file.writelines('EPISODE, TOTAL_REWARD, EPISOLON, STEP\n')

    memory = ReplayExperience(MEMORY_SIZE)
    start = 0.
    tot_reward = 0
    episode = 0
    count = 0
    epsilon = EPSILON_INITIAL
    epsilon_step = (EPSILON_INITIAL - EPSILON_FINAL) / EPSILON_STEPS

    for episode in range(NUM_EPISODES):
        env.reset()
        reward, done, state = 0, False, []
        for i in range(NUM_FRAMES):
            _observation, _reward, _done, _info =  env.step(0)
            reward += _reward
            state.append(image_preprocess(_observation))
            done = done | _done
        
        state_old = np.stack(state, axis=-1)
        
        tot_reward = 0
        while not done:
            start = time.time()

            if ENV_SHOW_TRAIN:
                # Render screen emulator
                env.render()

            # RL choose action based on observation of state or random action (exploitation or exploration)
            action, cat_action = tech.predict_action(state_old, epsilon)

            if epsilon > EPSILON_FINAL:
                epsilon -= epsilon_step

            # RL take action and get next state and reward
            reward, done, state = 0, False, []
            for i in range(NUM_FRAMES):
                # Take more frames to remove flickering
                _observation, _reward, _done, _info =  env.step(action)
                reward += _reward # accumulate reward
                state.append(image_preprocess(_observation)) # add pre-process frame to state
                done = done | _done
            state = np.stack(state, axis=-1)
            tot_reward += reward

            # Storing information to Replay experience
            memory.add(state_old, cat_action, reward, state, done)

            # Learning process
            if memory.count >= MEMORY_MIN_2_LEARN:                
                batch_state_old, batch_action, batch_reward, batch_state, batch_done = memory.sample(MEMORY_BATCH_SIZE)
                tech.learn(batch_state_old, batch_action, batch_reward, batch_state, batch_done)

                if tech.name == 'AtariDouble' or tech.name == 'AtariDueling':
                    if count % DQN_UPDATE_TARGET == 0:
                        #print ('update target model')
                        tech.target_model_update()
                

            # Save the network every 100000 iterations
            if count % DQN_SAVE_EVERY == 0:#if count % 100000 == 99999:
                print("Saving Network")
                tech.save_network(tech.name+'-'+str(count)+'-wights.h5')

            # Change old state
            state_old = state
            
            end = time.time()
        
            count += 1
        
        episode += 1
        print('EPISODE: {0:6d} | T_REWARD: {1:3.0f} | EPISOLON: {2:.6f} | STEP: {3:8d}'.format(
            episode, tot_reward, epsilon, count
        ))
        with open(tech.name+'_'+ENVIRONMENTS+'_episodes.csv', 'a') as file:
            file.write('{0:6d}, {1:3.0f}, {2:.6f}, {3:8d}\n'.format(
                episode, tot_reward, epsilon, count
            ))
    env.reset()
    env.close()
    env.render(close=True)

def simulate(tech):
    if SAVE_TEST:
        with open(WIGHTS+'_'+ENVIRONMENTS+'_'+str(IT)+'_episodes.csv', 'a') as file:
            file.writelines('0, 0, 0, 0\n')
    epsilon = 0
    start = 0.
    tot_reward = 0
    episode = 0
    count = 0

    for episode in range(NUM_EPISODES_AT_TEST):
        env.reset()
        reward, done, state = 0, False, []
        for i in range(NUM_FRAMES):
            _observation, _reward, _done, _info =  env.step(0)
            reward += _reward
            state.append(image_preprocess(_observation))
            done = done | _done
        
        state_old = np.stack(state, axis=-1)
        
        tot_reward = 0
        while not done:
            start = time.time()

            # RL choose action based on observation of state or random action (exploitation or exploration)
            action, cat_action = tech.predict_action(state_old, epsilon)

            # RL take action and get next state and reward
            reward, done, state = 0, False, []
            for i in range(NUM_FRAMES):
                # Take more frames to remove flickering
                _observation, _reward, _done, _info =  env.step(action)
                reward += _reward # accumulate reward
                state.append(image_preprocess(_observation)) # add pre-process frame to state
                done = done | _done

                if ENV_SHOW_TEST:
                    # Render screen emulator
                    env.render()
                    time.sleep(0.016)  

            state = np.stack(state, axis=-1)
            tot_reward += reward

          
            # Change old state
            state_old = state
            
            end = time.time()
        
            count += 1
        
        episode += 1
        print('EPISODE: {0:6d} | T_REWARD: {1:3.0f} | EPISOLON: {2:.6f} | STEP: {3:8d}'.format(
            episode, tot_reward, epsilon, count
        ))
        if SAVE_TEST:
            with open(WIGHTS+'_'+ENVIRONMENTS+'_'+str(IT)+'_episodes.csv', 'a') as file:
                file.write('{0:6d}, {1:3.0f}, {2:.6f}, {3:8d}\n'.format(
                    episode, tot_reward, epsilon, count
                ))

    env.reset()
    env.close()
    env.render(close=True)

def image_preprocess(obs, normalize=False):
    if not normalize:
        nor = 255
    else:
        nor = 1
        
    new_obs = resize(obs, (FRAME_WIDTH, FRAME_HEIGHT), mode='constant')
    if normalize:
        new_obs = np.array(rgb2gray(new_obs) * nor)
    else:
        new_obs = np.uint8(rgb2gray(new_obs) * nor)
    
    return new_obs

def print_obs(obs, gray=True):
    fig = plt.figure()
    if gray:
        plt.imshow(obs, interpolation='nearest', cmap='gray')
    else:
        plt.imshow(obs, interpolation='nearest')
    plt.show()



#lEnv = ['SpaceInvaders-v0', 'DemonAttack-v0']
lW = ['wights/Normal-AtariDueling-SpaceInvader-1000000-wights.h5', 
    'wights/Tf-AtariDueling-SpaceInvader-1000000-wights.h5', 
    'wights/Normal-AtariDueling-DemonAttack-1000000-wights.h5', 
    'wights/TF-AtariDueling-DemonAttack-1000000-wights.h5']

lEnv = ['DemonAttack-v0']
lW = ['wights/wights_demon_5M.h5']
for IT in range(2):
    for ENVIRONMENTS in lEnv:
        for WIGHTS in lW:

            ENV_SHOW_TRAIN = False
            ENV_SHOW_TEST = True
            SAVE_TEST = False
            
            env = gym.make(ENVIRONMENTS)
            NUM_ACTIONS = env.action_space.n

            random.seed(1)
            tech = AtariDueling()
            tech.load_network(WIGHTS)
            simulate(tech)
            #train(tech)
