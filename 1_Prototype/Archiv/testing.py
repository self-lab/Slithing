# 1) SliherMoves NxN Matrix per Move
# 2) Score Array of all Slithers
# 3) Life Status of all Slithers
# 4) Move Count

import random
import numpy as np
print(random.randrange(4))


dir = {0:'R',
       1:'L',
       2:'U',
       3:'D',
      }

print(dir[1])


a = np.zeros([9,9])

for i in range(9):
    if i//3-1>0:
        a[i-1,j]=1

round(7/3)
print(a)

def djkstr(matrix, m,n, start, target=4 default_val=200):
    cost = [[0 for x in range(m)] for x in range(1)]
    offsets = [start]
    elepos = 0
    path = [start]
    for j in range(m):
        cost[0][j] = matrix[start][j]
    for x in range(m-1):
        mini = default_val
        for j in range(m):                                                      #Select Minimum of next Node
            if cost[0][j]<=mini and j not in offsets:
                mini = cost[0][j]
                elepos = j
        offsets.append(elepos)
        for j in range(m):
            if cost[0][j] > cost[0][elepos] + matrix[elepos][j]:
                cost[0][j] = cost[0][elepos] + matrix[elepos][j]
    print('The shortest path', offsets)
    print('The cost to various vertices in order', cost)


def main():
    print('Dijkstras algorithm graph using matrix representation')

    a = [[0,1,100,100,100],
         [100,0,100,100,1],
         [100,100,0,100,100],
         [100,5,6,0,100],
         [100,100,100,3,0]]
    a
    djkstr(a,len(a),len(a[0]), start=0)

main()
