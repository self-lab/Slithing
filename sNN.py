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
        self.memory = Memory(500000)
        self.STORE_PATH = 'C:/Users/Milan/Desktop/Coding/Python/Projects/1_Slithing/CheckPoints'
        self.MAX_EPSILON = 1
        self.MIN_EPSILON = 0
        self.LAMBDA = 0.0005
        self.GAMMA = 0.95
        self.BATCH_SIZE = 2
        self.TAU = 0.08
        self.RANDOM_REWARD_STD = 1.0
        self.train_writer = tf.summary.create_file_writer(self.STORE_PATH + f"/DoubleQ_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
        self.state_size = 484
        self.num_actions = 4
        self.slither = slith

        self.primary_network = keras.Sequential([
            keras.layers.Dense(484, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            keras.layers.Dense(484, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            # keras.layers.Dense(484, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            keras.layers.Dense(self.num_actions)
        ])

        self.target_network = keras.Sequential([
            keras.layers.Dense(484, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            keras.layers.Dense(484, activation='relu', kernel_initializer=keras.initializers.he_normal()),
    #        keras.layers.Dense(484, activation='relu', kernel_initializer=keras.initializers.he_normal()),
            keras.layers.Dense(self.num_actions)
        ])

        self.primary_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')




    def choose_action(self, state, primary_network, eps):
        if random.random() < eps:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.primary_network(state.to_numpy().reshape(1, -1)))



    def train(self, primary_network, memory, target_network=None):
        print(memory.num_samples, self.BATCH_SIZE)
        if memory.num_samples < self.BATCH_SIZE * 3:
            return 0
        batch = memory.sample(self.BATCH_SIZE)
        #states = np.array([val[0] for val in batch])
        states = np.array([item[0].to_numpy().reshape(1,-1)[0] for item in batch])
        actions = np.array([val[1] for val in batch])
        rewards = np.array([val[2] for val in batch])
        # next_states = np.array([(np.zeros(self.state_size)
        #                          if val[3] is None else val[3]) for val in batch])

        next_states = np.array([(np.zeros(self.state_size)
                                 if item[3] is None else item[3].to_numpy().reshape(1,-1)[0]) for item in batch])


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
            updates[valid_idxs] += self.GAMMA * np.amax(prim_qtp1.numpy()[valid_idxs, :], axis=1)
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


    def run_training(self):
        num_episodes = 500
        eps = self.MAX_EPSILON
        render = True
        self.train_writer = tf.summary.create_file_writer(self.STORE_PATH + f"/DoubleQ_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
        double_q = True
        steps = 0
        for i in range(num_episodes):
            state = 'changeme'
            cnt = 0
            avg_loss = 0
            while True:
                action = self.choose_action(state, self.primary_network, eps)
                next_state, reward, done, info = env.step(action)
                reward = np.random.normal(1.0, self.RANDOM_REWARD_STD)
                if done:
                    next_state = None
                # store in memory
                memory.add_sample((state, action, reward, next_state))

                loss = train(self.primary_network, memory, self.target_network if double_q else None)
                avg_loss += loss

                state = next_state

                # exponentially decay the eps value
                steps += 1
                eps = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * math.exp(-self.LAMBDA * steps)

                if done:
                    avg_loss /= cnt
                    print(f"Episode: {i}, Reward: {cnt}, avg loss: {avg_loss:.3f}, eps: {eps:.3f}")
                    with train_writer.as_default():
                        tf.summary.scalar('reward', cnt, step=i)
                        tf.summary.scalar('avg loss', avg_loss, step=i)
                    break

                cnt += 1
