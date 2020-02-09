import tensorflow as tf
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
from tensorflow import keras
import numpy as np
import pandas as pd
from random import randrange
import random
import math
import datetime as dt

class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)

    @property
    def num_samples(self):
        return len(self._samples)


class sNN():
    def __init__(self, slith=0):
        self.memory = Memory(10000000)
        self.STORE_PATH = 'C:/Users/Milan/Desktop/Coding/Python/Projects/1_Slithing/CheckPoints'
        self.MAX_EPSILON = 1
        self.MIN_EPSILON = 0.001
        self.LAMBDA = 0.005
        self.GAMMA = 0.7
        self.BATCH_SIZE = 500
        self.TAU = 0.08
        self.RANDOM_REWARD_STD = 1.0
        self.train_writer = tf.summary.create_file_writer(self.STORE_PATH + f"/DoubleQ_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
        self.state_size = 49
        self.num_actions = 4
        self.slither = slith                                                    # Can probly remove soon
        self.THRESHOLD = 1
        keras.backend.set_floatx('float64')

        self.primary_network = keras.Sequential([
            keras.layers.Dense(self.state_size, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            # keras.layers.Dense(self.state_size, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            # keras.layers.Dense(self.state_size, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            # keras.layers.Dense(self.state_size, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            # keras.layers.Dense(self.state_size, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            # keras.layers.Dense(self.state_size, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            # keras.layers.Dense(self.state_size, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            # keras.layers.Dense(self.state_size, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            # keras.layers.Dense(self.state_size, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            # keras.layers.Dense(self.state_size, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            keras.layers.Dense(self.num_actions)
        ])

        self.target_network = None #keras.Sequential([
        #     keras.layers.Dense(self.state_size, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        #     keras.layers.Dense(self.state_size, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        #     keras.layers.Dense(self.state_size, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        #     keras.layers.Dense(self.state_size, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        #     keras.layers.Dense(self.num_actions)
        # ])

        self.primary_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')

        self.direction = {
          0: 'up',
          1: 'down',
          2: 'left',
          3: 'right'
        }

    def choose_action(self, state, primary_network, eps):
        if random.random() < min(eps, self.THRESHOLD):
            return random.randint(0, self.num_actions - 1)
        else:
            direction = np.argmax(self.primary_network(state.to_numpy().reshape(1, -1)))
            return direction

    def train(self, primary_network, memory, target_network=None):
        if memory.num_samples < self.BATCH_SIZE * 2:
            return 0
        batch = memory.sample(self.BATCH_SIZE)
        states = np.array([item[0].to_numpy().reshape(1,-1)[0] for item in batch])
        actions = np.array([val[1] for val in batch])
        rewards = np.array([val[2] for val in batch])


        next_states = np.array([(np.zeros(self.state_size)
                                 if item[3] is None else item[3].to_numpy().reshape(1,-1)[0]) for item in batch])

        primary_network.summary()

        # predict Q(s,a) given the batch of states
        #print(states.shape)
        prim_qt = primary_network(states)

        # predict Q(s',a') from the evaluation network
        prim_qtp1 = primary_network(next_states)

        # copy the prim_qt tensor into the target_q tensor - we then will update one index corresponding to the max action
        target_q = prim_qt.numpy()
        updates = rewards
        valid_idxs = np.array(next_states).sum(axis=1) != 0
        batch_idxs = np.arange(self.BATCH_SIZE)
        if target_network is None:
            updates[valid_idxs] =np.add(updates[valid_idxs], self.GAMMA * np.amax(prim_qtp1.numpy()[valid_idxs, :], axis=1), casting='unsafe')
        else:
            prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
            q_from_target = target_network(next_states)
            updates[valid_idxs]= np.add(updates[valid_idxs] , self.GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs], prim_action_tp1[valid_idxs]], casting='unsafe')
        target_q[batch_idxs, actions] = updates
        loss = primary_network.train_on_batch(states, target_q)
        if target_network is not None:
            # update target network parameters slowly from primary network
            for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
                t.assign(t * (1 - self.TAU) + e * self.TAU)
        return loss
