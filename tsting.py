import pandas as pd
import numpy as np

a = [x for x in range(5)]

a = pd.DataFrame(np.eye(12))

rotate = pd.DataFrame(np.matrix(
[[1,1,1,1,1,1,1,1,1,1,1,1],
 [1,0,0,0,0,0,0,0,0,3,0,1],
 [1,0,0,0,0,0,0,4,0,0,0,1],
 [1,0,3,0,0,0,0,2,0,0,0,1],
 [1,0,0,0,0,0,2,2,0,0,0,1],
 [1,0,0,0,0,0,0,0,0,0,0,1],
 [1,0,0,0,0,0,0,0,0,0,0,1],
 [1,0,0,0,0,0,0,0,0,0,0,1],
 [1,0,0,0,0,0,0,0,3,0,0,1],
 [1,0,0,0,0,0,0,0,0,0,0,1],
 [1,1,1,1,1,1,1,1,1,1,1,1]]
))


def myview(field, pos, rad, direction = 0):
    xlbound = max(pos[0]-rad,0)
    xubound = max(pos[0]+rad+1,0)
    ylbound = max(pos[1]-rad,0)
    yubound = max(pos[1]+rad+1,0)

    if pos[0]+rad+1>field.shape[0]:
        x_offset  = -(pos[0] + rad + 1 - field.shape[0])
    elif pos[0] - rad < 0:
        x_offset = -(pos[0] - rad)
    else:
        x_offset = None

    if pos[1]+rad+1>field.shape[1]:
        y_offset  = -(pos[1] + rad + 1 - field.shape[1])
    elif pos[1] - rad < 0:
        y_offset = -(pos[1] - rad)
    else:
        y_offset = None

    #print('x_offset: ', x_offset,'y_offset: ', y_offset)

    tmp = pd.DataFrame(np.ones((2*rad+1,2*rad+1)))

    tmp.iloc[x_offset if x_offset > 0 else None : x_offset if x_offset < 0 else None,
             y_offset if y_offset > 0 else None : y_offset if y_offset < 0 else None]\
             = field.iloc[xlbound:xubound, ylbound:yubound].values

    if direction == 0:
        pass
    if direction == 1:
        tmp = np.rot90(tmp,2)
    if direction == 2:
        tmp = np.rot90(tmp,3)
    if direction == 3:
        tmp = np.rot90(tmp,1)

    return pd.DataFrame(tmp)
    # return(field.iloc[xlbound:xubound,
    #                    ylbound:yubound])

rotate.iloc[0:8, 2:13].shape

tmp = myview(rotate,[2,7], 5, direction=3)
tmp

tmp.iloc[:,:-1]
print(np.rot90(tmp,4))

np.rot90(res.iloc[0:,3:11])
tst = pd.DataFrame(np.ones((4,4)))

tst.iloc[0:2,0:2]=tmp.iloc[0:2,0:2].values
tmp.iloc[0:2,0:2]
tst.shape[0]
rotate

if 'a'.ischar():
    print('fml')
