from os import replace
from numpy.lib.ufunclike import _fix_out_named_y
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np

class DDQN(keras.Model):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DDQN, self).__init__()
        assert input_dims == [8,]
        self.dense0 = tf.keras.layers.Dense(input_dims[0], activation='relu')
        self.dense1 = tf.keras.layers.Dense(fc1_dims, activation ='relu')
        self.dense2 = tf.keras.layers.Dense(fc2_dims, activation = 'relu')
        self.V = tf.keras.layers.Dense(1, activation = None)
        self.A = tf.keras.layers.Dense(n_actions, activation= None)

    def call(self, state):
        x = self.dense0(state)
        y = self.dense1(x)
        z = self.dense2(y)
        V = self.V(z)
        A = self.A(z)
        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims =True))) 

        return Q 

    def advantage(self, state):
        x = self.dense0(state)
        y = self.dense1(x)
        z = self.dense2(y)
        A = self.A(z)
        return A
     
class RepBuff:
    # store the state transitions --- so we initially set up numpy arrays of a set size.
    # one for each: state, action, reward, newstate, done...
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)
    
    # store each transition....
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    # function to sample a batch of a set size (batch_size) from the replay buffer... 
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
    
        # store parameters that have been passed in
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.fname = fname 
        self.replace = replace
        self.batch_size = batch_size

        # initialise the step-counter, the memory, set up and compile both networks...
        self.learn_step_counter = 0
        self.memory  = RepBuff(mem_size, input_dims)
        self.q_eval = DDQN(input_dims, fc1_dims, fc2_dms, n_actions)
        self.q_next = DDQN(input_dims, fc1_dims, fc2_dms, n_actions)
        self.q_eval.compile(optimizer=Adam(learning_rate=lr),loss='mean_squared_error')
        self.q_next.compile(optimizer=Adam(learning_rate=lr),loss='mean_squared_error')
        
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
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
        if self.memory.mem_cntr < self.batch_size:
            # if we haven't enough observations to train on yet
            return 
        # assuming we have enough memory...
        # update/transfer weights from one network to another network based on hyperparams...
        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())
        # DRAW A BATCH OF SIZE     
        states, actions, rewards, new_states, dones = \
            self.memory.sample_buffer(self.batch_size)
        q_pred = self.q_eval(states)
        q_next = tf.math.reduce_max(self.q_next(new_states),axis =1, keepdims=True).numpy()
        q_target = np.copy(q_pred)

        for idx, terminal in enumerate(dones):
            if terminal:
                q_next[idx] = 0.0
            q_target[idx,actions[idx]] = rewards[idx] + self.gamma *q_next[idx]
        self.q_eval.train_on_batch(states, q_target)
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_end else self.epsilon_end
        self.learn_step_counter += 1

    def save_model(self):
        self.q_eval.save(self.fname)
    
    def load_model(self):
        self.q_eval = load_model(self.fname)




            




