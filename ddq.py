from os import replace
from numpy.lib.ufunclike import _fix_out_named_y
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np

class DDQN(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions):
        super(DuellingDeepQNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(fc1_dims, activation ='relu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation = 'relu')
        self.V = keras.layers.Dense(1, actvation = None)
        self.A = keras.layers.Dense(n_actions, activation= None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims =True)))

        return Q 

    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(state)
        A = self.A(x)
        return A
     
class RepBuff:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)
    
    def store_tranistion(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones
        


class Agent:
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec = 1e-3, epsilon_end = 0.01, mem_size = 100000, fname = 'ddq_DQN.h5', fc1_dims = 128, fc2_dms = 124, replace = 100):
    
        # store parameters
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.fname = fname 
        self.replace = replace

        # hyperparams
        self.learn_step_counter = 0
        self.memory  = RepBuff(mem_size, input_dims)
        self.q_eval = DDQN(fc1_dims, fc2_dms, n_actions)
        self.q_next = DDQN(fc1_dims, fc2_dms, n_actions)
        self.q_eval.compile(optimizer=Adam(learning_rate=lr),loss='mean_squared_error')
        self.q_next.compile(optimizer=Adam(learning_rate=lr),loss='mean_squared_error')
        
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_tranistion(state, action, reward, new_state, done)
    
    def choose_action(self, observation):
        # epsilon greedy
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0] # what is the point of this line? 

        return action
    
    def learn(self):
        # TODO: COMPLETE BY TOMORROW EOD. 






