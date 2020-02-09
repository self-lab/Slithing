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
import sNN as sai


print(tf.__version__)

class prototype():
    def __init__(self):
        self.previous_state = None
        self.arg_score = 0
        self.neural_network = sai.sNN()
        self.move = 1
        self.total_moves = 0
        self.eps = self.neural_network.MAX_EPSILON
        self.avg_loss = 0
        self.counter = 0

        self.direction = {
          0: 'up',
          1: 'down',
          2: 'left',
          3: 'right'
        }


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
        if matrix[i,j] == str(0) or matrix[i,j] == 3 or matrix[i,j]==0 or matrix[i,j] in range(4,20):
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
                if (x-1) > 0 and self.valid_path(matrix, x-1,y):                # Snake goes up
                    adj_mat[i,self.tuple_to_adj(x-1,y,lcol=lcol)] = 1
                if (x+1) <= lrow and self.valid_path(matrix, x+1,y):            # Snake goes down
                    adj_mat[i,self.tuple_to_adj(x+1,y,lcol=lcol)] = 1
                if (y-1) > 0 and self.valid_path(matrix, x,y-1):                # Snake goes left
                    adj_mat[i, self.tuple_to_adj(x,y-1,lcol=lcol)] = 1
                if (y+1) <= lcol and self.valid_path(matrix, x,y+1):            # Snake goes right
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
        row = col = len(adj)                                                    # Load Dimensions (adj = symmetric)
        dist = [np.inf]*row                                                     # Initialize Distance array to infinity
        path = [-1]*row                                                         # Initialize Path Array to some dummy variable
        dist[start] = 0                                                         # From the starting point, the distance is zero

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
            return 0                                                            # Up
        elif nx > cx:
            return 1                                                            # Down
        elif ny > cy:
            return 3                                                            # Right
        elif ny < cy:
            return 2                                                            # Left
        else:
            pass

    def train_slither(self, state, action, reward, next_state,
                      slith_alive, slith_score):
        self.move += 1
        self.neural_network.memory.add_sample(
            (
            state,
            action,
            reward,
            next_state
            )
        )
        self.loss =\
            self.neural_network.train(
                self.neural_network.primary_network,
                self.neural_network.memory,
                self.neural_network.target_network)
        self.avg_loss += self.loss

        # if reward == 5:
        #     print('State: ', state,'\n\nAction: ', action,'\n\nnext_state', next_state,'\nLoss:', loss)
        #     import time
        #     time.sleep(10)

        self.eps =\
            self.neural_network.MIN_EPSILON \
            + (self.neural_network.MAX_EPSILON
            - self.neural_network.MIN_EPSILON)\
            * math.exp(- self.neural_network.LAMBDA * self.total_moves)

        if not slith_alive:
            self.print_status(slith_score)

    def print_status(self, slith_score):
        if self.loss > 0:
            self.counter +=1
        print(f"Episode: {self.counter}, Reward: {slith_score}, 'Moves': {self.move}, avg loss: {self.avg_loss/self.move:.3f}, eps: {self.eps:.3f}")
        self.total_moves += self.move
        self.move = 1
        self.avg_loss = 0
