
'''
Inspired:
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/
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

class DeepQNetwork():
    def __init__(
        self, 
        actionsList,
        nFeatures,
        learningRate=0.01, 
        rewardDecay=0.95, 
        eGreedy=0.95, 
        replaceTargetIter=300,
        memorySize=500,
        batchSize=32,
        eGreedyIncrement=None,
        outputGraph=False):
        
        self.actionsList = actionsList
        self.nFeatures = nFeatures
        self.lr = learningRate
        self.gamma = rewardDecay
        self.epsilon_max = eGreedy
        self.replace_target_iter = replaceTargetIter
        self.memory_size = memorySize
        self.batch_size = batchSize
        self.epsilon_increment = eGreedyIncrement
        self.epsilon = 0 if eGreedyIncrement is not None else self.epsilon_max
        # Total learning step
        self.learn_step_counter = 0

        # Initialize memory
        self.memory = np.zeros((self.memory_size, nFeatures*2 +2))

        # Initialize model
        self._build_net()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if outputGraph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his= []
    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.nFeatures], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.nFeatures], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, len(self.actionsList), kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, len(self.actionsList), kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
    def choose_action(self, state):
        # to have batch dimension when feed into tf placeholder
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every action
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: state})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0,len(self.actionsList))

        return action
        
    def storeTransition(self, state, action, reward, state_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = np.hstack((state, [action, reward], state_))
        #replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def plotCost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.nFeatures],
                self.a: batch_memory[:, self.nFeatures],
                self.r: batch_memory[:, self.nFeatures + 1],
                self.s_: batch_memory[:, -self.nFeatures:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

class DeepQNetworkKeras():
    def __init__(
        self, 
        actionsList,
        nFeatures,
        learningRate=0.01, 
        rewardDecay=0.95, 
        eGreedy=0.95, 
        replaceTargetIter=300,
        memorySize=500,
        batchSize=32,
        eGreedyIncrement=None,
        outputGraph=False,
        layer1Units=50,
        layer2Units=100,
        input_dim=2,
        epochs=1, 
        batch_size=20):
        
        self.actionsList = actionsList
        self.nFeatures = nFeatures
        self.lr = learningRate
        self.gamma = rewardDecay
        self.epsilon_max = eGreedy
        self.replace_target_iter = replaceTargetIter
        self.memory_size = memorySize
        self.batch_size = batchSize
        self.epsilon_increment = eGreedyIncrement
        self.epsilon = 0 if eGreedyIncrement is not None else self.epsilon_max
        self.layer1Units = layer1Units
        self.layer2Units = layer2Units
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size

        # Total learning step
        self.learn_step_counter = 0

        # Initialize memory
        self.memory = np.zeros((self.memory_size, 4), dtype='object')

        # Initialize model
        self.model = self._build_net()
       

    def _build_net(self):
        model = Sequential()
        model.add(Dense(self.layer1Units, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(self.layer2Units, activation='relu'))
        model.add(Dense(len(self.actionsList)))
        adam = Adam(lr = self.lr)
        model.compile(optimizer = adam, loss='mean_squared_error')
        
        return model

    def storeTransition(self, state, action, reward, state_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = (state.T, action, reward, state_.T)
        #replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            actions_value = self.model.predict(np.expand_dims(state, axis=0))
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, len(self.actionsList))
        
        return action

    def _get_data(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        if len(batch_memory) > 0:
            env_size = batch_memory[0][0].shape[0]
            mem_size = len(batch_memory)

            inputs = np.zeros((mem_size, env_size))
            targets = np.zeros((mem_size, len(self.actionsList)))
            i = 0
            for state, action, reward, state_ in batch_memory:
                inputs[i] = state
                targets[i] = self.model.predict(np.expand_dims(state, axis=0))

                sa = self.model.predict(np.expand_dims(state_, axis=0))
                Q_sa = np.max(sa)

                targets[i, action] = reward + self.gamma * Q_sa
                i += 1
            return inputs, targets
        return None, None


    def learn(self):
        inputs, targets = self._get_data()
        
        historic = self.model.fit(inputs, targets, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max


    def plotCost(self):
        pass

class DoubleDeepQNetworkKeras():
    def __init__(
        self, 
        actionsList,
        nFeatures,
        learningRate=0.01, 
        rewardDecay=0.95, 
        eGreedy=0.95, 
        replaceTargetIter=300,
        memorySize=500,
        batchSize=32,
        eGreedyIncrement=None,
        outputGraph=False,
        layer1Units=50,
        layer2Units=100,
        input_dim=2,
        epochs=1, 
        batch_size=20,
        tau=0.001):
        
        self.actionsList = actionsList
        self.nFeatures = nFeatures
        self.lr = learningRate
        self.gamma = rewardDecay
        self.epsilon_max = eGreedy
        self.replace_target_iter = replaceTargetIter
        self.memory_size = memorySize
        self.batch_size = batchSize
        self.epsilon_increment = eGreedyIncrement
        self.epsilon = 0 if eGreedyIncrement is not None else self.epsilon_max
        self.layer1Units = layer1Units
        self.layer2Units = layer2Units
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.tau = tau

        # Total learning step
        self.learn_step_counter = 0

        # Initialize memory
        self.memory = np.zeros((self.memory_size, 4), dtype='object')

        # Initialize model
        self.model = self._build_dual_net()
        self.target_model = self._build_dual_net()
        self.update_target_model(self.tau)
       

    def _build_net(self):
        model = Sequential()
        model.add(Dense(self.layer1Units, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(self.layer2Units, activation='relu'))
        model.add(Dense(len(self.actionsList)))
        adam = Adam(lr = self.lr)
        model.compile(optimizer = adam, loss='mean_squared_error')
        
        return model

    def _build_dual_net(self):
        model = Sequential()
        input_layer = Input(shape=(self.input_dim,))
        
        fc1 = Dense(self.layer1Units, activation='relu')(input_layer)
        advantage = Dense(len(self.actionsList))(fc1)
        
        fc2 = Dense(self.layer1Units, activation='relu')(input_layer)
        value = Dense(1)(fc2)

        policy = merge([advantage, value], 
            mode = lambda x: x[0] - K.mean(x[0])+x[1],
            output_shape = (len(self.actionsList),)
            )

        model = Model(input=[input_layer], output=[policy])
        adam = Adam(lr = self.lr)
        model.compile(optimizer = adam, loss='mean_squared_error')
        
        return model

    def update_target_model(self, tau):
        if tau == 1:
            self.target_model.set_weights(self.model.get_weights())
        else:
            main_weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            for i, layer_weights in enumerate(main_weights):
                target_weights[i] *= (1-tau)
                target_weights[i] += tau * layer_weights
            self.target_model.set_weights(target_weights)

    def storeTransition(self, state, action, reward, state_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = (state.T, action, reward, state_.T)
        #replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            actions_value = self.model.predict(np.expand_dims(state, axis=0))
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, len(self.actionsList))
        
        return action

    def _get_data(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        if len(batch_memory) > 0:
            env_size = batch_memory[0][0].shape[0]
            mem_size = len(batch_memory)

            inputs = np.zeros((mem_size, env_size))
            targets = np.zeros((mem_size, len(self.actionsList)))
            i = 0
            for state, action, reward, state_ in batch_memory:
                inputs[i] = state
                targets[i] = self.model.predict(np.expand_dims(state, axis=0))

                sa = self.target_model.predict(np.expand_dims(state_, axis=0))
                Q_sa = np.max(sa)

                targets[i, action] = reward + self.gamma * Q_sa
                i += 1
            return inputs, targets
        return None, None


    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.update_target_model(self.tau)
            print('\ntarget_params_replaced\n')
        
        inputs, targets = self._get_data()
        historic = self.model.fit(inputs, targets, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self.learn_step_counter += 1


    def plotCost(self):
        pass