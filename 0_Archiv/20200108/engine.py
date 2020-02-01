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


print(tf.__version__)

class prototype():
    def __init__(self):
        pass

    def set_action(self, Field, Head, Food):
        tmpA = self.create_adjmatrix(np.array(Field))
        numb_nodes = len(Field)-2
        spos = self.tuple_to_adj(Head[0], Head[1], numb_nodes)
        dist, path, queue = self.dijkstra(tmpA, start=spos)
        adjFood = self.tuple_to_adj(Food[0][0], Food[0][1], numb_nodes)
        spath = self.get_path(path, adjFood)
        spath2 = self.makeSlithpath(spath, numb_nodes)
        return self.djkstra_path(spath2, Head)

    def tuple_to_adj(self, i, j, lcol=3):
        return (i-1)*lcol + (j-1)

    def numb_to_tuple(self, n, lcol=3):
        return (math.ceil(n//lcol)+1), n%lcol +1

    def valid_path(self, matrix, i, j):
        if matrix[i,j] == str(0) or matrix[i,j] == 'F' or matrix[i,j]==0 or matrix[i,j] == 'H':
            return True
        else:
            return False

    def create_adjmatrix(self, matrix):
        lrow, lcol = matrix.shape
        lrow -= 2
        lcol -= 2
        ladjm = lrow*lcol
        adj_mat = np.full([ladjm, ladjm], np.inf)
        #return lrow, lcol
        for i in range(ladjm):
            x,y = self.numb_to_tuple(i, lcol=lcol)
            adj_mat[i,i]=0
            if self.valid_path(matrix,x,y):
            #1 if up, down, right or left movement is possible
                if (x-1) > 0 and self.valid_path(matrix, x-1,y):                #Snake goes up
                    adj_mat[i,self.tuple_to_adj(x-1,y,lcol=lcol)] = 1
                if (x+1) <= lrow and self.valid_path(matrix, x+1,y):            #Snake goes down
                    adj_mat[i,self.tuple_to_adj(x+1,y,lcol=lcol)] = 1
                if (y-1) > 0 and self.valid_path(matrix, x,y-1):                #Snake goes left
                    adj_mat[i, self.tuple_to_adj(x,y-1,lcol=lcol)] = 1
                if (y+1) <= lcol and self.valid_path(matrix, x,y+1):            #Snake goes right
                    adj_mat[i, self.tuple_to_adj(x,y+1,lcol=lcol)] = 1

        return adj_mat



    def minDistance(self, dist, queue):
        # Initialize min value and min_index as -1
        minimum = np.inf
        min_index = -1

        for i in range(len(dist)):
            if dist[i] < minimum and i in queue:
                minimum = dist[i]
                min_index = i

        if min_index != -1:
            return min_index
        else:
            return queue[0]

    def dijkstra(self, adj, start = 0):
        row = col = len(adj)                                                    #Load Dimensions (adj = symmetric)
        dist = [np.inf]*row                                                     #Initialize Distance array to infinity
        path = [-1]*row                                                         #Initialize Path Array to some dummy variable
        dist[start] = 0                                                         #From the starting point, the distance is zero

        queue = [x for x in range(row)]

        while queue:
            ind = self.minDistance(dist, queue)
            queue.remove(ind)
            for i in range(col):
                if adj[ind][i] != np.inf and i in queue:
                    if dist[ind] + adj[ind][i] < dist[i]:
                        dist[i] = dist[ind] + adj[ind][i]
                        path[i] = ind


        return dist, path, queue



    def get_path(self, path, dest):
        slither_path = []
        temp = dest
        while temp != -1:
            slither_path.insert(0,temp)
            temp = path[temp]
        return slither_path


    def makeSlithpath(self, path, col):
        slith_path = []
        for i in range(len(path)):
            slith_path.append(self.numb_to_tuple(path[i], col))

        return slith_path

    def djkstra_path(self, slith_path, Coords):
        if len(slith_path)>1:
            next = list(slith_path[1])
        else:
            return randrange(4)
        current = Coords
        cx = Coords[0]
        cy = Coords[1]
        nx = next[0]
        ny = next[1]

        if nx < cx:
            return 0                                                            #Up
        elif nx > cx:
            return 1                                                            #Down
        elif ny > cy:
            return 3                                                            #Right
        elif ny < cy:
            return 2                                                            #Left
        else:
            pass

########-----------------------AI-TESTING PHASE---------------------############

if __name__=='__main__':
    print('wtf?')
    STORE_PATH = 'C:/Users/Milan/Desktop/Coding/Python/Projects/1_Slithing'
    MAX_EPSILON = 1
    MIN_EPSILON = 0.01
    LAMBDA = 0.0005
    GAMMA = 0.95
    BATCH_SIZE = 32
    TAU = 0.08
    RANDOM_REWARD_STD = 1.0
    state_size = 20*20
    num_actions = 4

    primary_network = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(num_actions)
    ])

    target_network = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(num_actions)
    ])

    primary_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')


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


    memory = Memory(500000)

    def choose_action(state, primary_network, eps):
        if random.random() < eps:
            return random.randint(0, num_actions - 1)
        else:
            return np.argmax(primary_network(state.reshape(1, -1)))

    def train(primary_network, memory, target_network=None):
        if memory.num_samples < BATCH_SIZE * 3:
            return 0
        batch = memory.sample(BATCH_SIZE)
        states = np.array([val[0] for val in batch])
        actions = np.array([val[1] for val in batch])
        rewards = np.array([val[2] for val in batch])
        next_states = np.array([(np.zeros(state_size)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        print(states)
        prim_qt = primary_network(states)
        # predict Q(s',a') from the evaluation network
        prim_qtp1 = primary_network(next_states)
        # copy the prim_qt tensor into the target_q tensor - we then will update one index corresponding to the max action
        target_q = prim_qt.numpy()
        updates = rewards
        valid_idxs = np.array(next_states).sum(axis=1) != 0
        batch_idxs = np.arange(BATCH_SIZE)
        if target_network is None:
            updates[valid_idxs] += GAMMA * np.amax(prim_qtp1.numpy()[valid_idxs, :], axis=1)
        else:
            prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
            q_from_target = target_network(next_states)
            updates[valid_idxs] += GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs], prim_action_tp1[valid_idxs]]
        target_q[batch_idxs, actions] = updates
        loss = primary_network.train_on_batch(states, target_q)
        if target_network is not None:
            # update target network parameters slowly from primary network
            for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
                t.assign(t * (1 - TAU) + e * TAU)
        return loss

    ####------------------TRAINING-------------###
    num_episodes = 500
    eps = MAX_EPSILON
    render = True
    train_writer = tf.summary.create_file_writer(STORE_PATH + f"/DoubleQ_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
    double_q = True
    steps = 0
    for i in range(num_episodes):
        state = 'changeme'
        cnt = 0
        avg_loss = 0
        while True:
            # if render:
            #     env.render()
            action = choose_action(state, primary_network, eps)
            next_state, reward, done, info = env.step(action)
            reward = np.random.normal(1.0, RANDOM_REWARD_STD)
            if done:
                next_state = None
            # store in memory
            memory.add_sample((state, action, reward, next_state))

            loss = train(primary_network, memory, target_network if double_q else None)
            avg_loss += loss

            state = next_state

            # exponentially decay the eps value
            steps += 1
            eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps)

            if done:
                avg_loss /= cnt
                print(f"Episode: {i}, Reward: {cnt}, avg loss: {avg_loss:.3f}, eps: {eps:.3f}")
                with train_writer.as_default():
                    tf.summary.scalar('reward', cnt, step=i)
                    tf.summary.scalar('avg loss', avg_loss, step=i)
                break

            cnt += 1

    tf.saved_model.save(primary_network, 'C:/Users/Milan/Desktop/Coding/Python/Projects/1_Slithing/model/')
