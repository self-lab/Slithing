'''
Created on Sun May 26 16:49:51 2019
@author: Milan Reichmann
TO DO Clean up and Reward Function to +10 if slither finds food
and -100 if slither rips
'''

#Standard Libraries
import sys                                                                      #Systemfuncitons, such as exit Programm
import random
import math
from random import randrange                                                    #Random N. Generator, i.e. Food Placement
import numpy as np
import pandas as pd
from math import floor
import time
#import tensorflow as tf

#Third Party Libraries - Professional
from PyQt5 import QtGui, QtWidgets, QtCore                                      #API für GuI, Widgets

#Third Party Libraries - Amateur
import TableView as myDataTable

#User defined
import engine
import sNN as sai


#Colors
#PwC Colors
qlightbrown = QtGui.QColor(208,74,2)                                            #6 Colors Required
qorange = QtGui.QColor(235,140,0)                                               #5 Colors Required
qstripe = QtGui.QColor(222,222,222)
#primariy Colors (Use these!)
qyellow = QtGui.QColor(255,182,0)
qpink = QtGui.QColor(219,83,106)
qred = QtGui.QColor(224,48,30)
qgrey = QtGui.QColor(125,125,125)
qdark = QtGui.QColor(70,70,70)



class slitherMember():
    def __init__(self, predecessor, x = None, y = None, Last=True, Pos = None):
        '''This class describes a Member other than the Slither Head. Any
        Member knows it's predecessors, (x,y) Coordinates, and it's previous
        Coordinates.'''
        self.predecessor = predecessor                                          #Every Member knows it's predecessor
        self.x = x
        self.y = y
        self.Pos = self.predecessor.returnPosition() + 1                        #Every Members knows it's Position in the array
        self.prevCords = (None,None)                                            #Every Member knows it's previous turns Coords
        self.Last = Last


    def returnCords(self):
        '''This function returns the Slithers (x,y)-Coordinates'''
        return self.x, self.y


    def returnPosx(self):
        '''This Function returns the Slither-Members X-Coordinates'''
        return self.x


    def returnPosy(self):
        '''This function returns the Slither-Members Y-Coordinates'''
        return self.y


    def returnPosition(self):
        '''This function returns the Slither-Members Position within the Slither
        Array'''
        return self.Pos


    def updateCords(self):
        '''This function updates the Slither Members coordinates, by setting
        them to the predecessors coordinates. Furthermore the previous turns
        Coordinates get updated'''
        self.prevCords = (self.x, self.y)
        self.x, self.y = self.predecessor.returnCords()


    def returnLastPos(self):
        '''This function returns the Slither Members previous Coordinates'''
        return self.prevCords[0], self.prevCords[1]


    def iwascalled(self):
        '''This function serves purely for debugging purposes'''
        print("(", self.x, ",", self.y,")", '  Position: ', self.Pos)


class slitherHead():
    def __init__(self,x,y, lastkey):
        '''Any Slitherhead knows it's Position, last direction and the last
        directional key that was pressed.'''
        self.x = x
        self.y = y
        self.direction = 'R'
        self.lastkey = lastkey


    def returnCords(self):
        '''Returns the Slithers Coordinates'''
        return self.x, self.y


    def returnPosx(self):
        '''Returns the SlitherHead's y Coordinate'''
        return self.x


    def returnPosy(self):
        '''Returns the SlitherHead's y Coordinate'''
        return self.y


    def returnPosition(self):
        '''This function returns the Heads Position, which by definition is 0.
        Note, that this function is overloaded, i.e. Members have the same
        function call'''
        return 0


    def iwascalled(self):
        '''This function is purely used for debugging purposes'''
        print(self.x, ",", self.y)


    def turnUp(self):
        '''This function ensures that no invalid movement occures by comparing
        to the last direction the Slither Moved.'''
        if self.lastkey != 'D':
            self.direction = 'U'


    def turnLeft(self):
        '''This function ensures that no invalid movement occures by comparing
        to the last direction the Slither Moved.'''
        if self.lastkey != 'R':
            self.direction = 'L'


    def turnRight(self):
        '''This function ensures that no invalid movement occures by comparing
        to the last direction the Slither Moved.'''
        if self.lastkey != 'L':
            self.direction = 'R'


    def turnDown(self):
        '''This function ensures that no invalid movement occures by comparing
        to the last direction the Slither Moved.'''
        if self.lastkey != 'U':
            self.direction = 'D'


    def updateCords(self, newx, newy, lk):
        '''Every Head knows it's own coordinates. After moving into one
        direction this function get's called to set the coordinates anew'''
        self.x = newx
        self.y = newy
        self.lastkey = lk


class Slither(slitherHead):
    def __init__(self, x, y,                                                    #x, y starting coordinates as Qt coordinates
                 slitherSize,                                                   #A Slither knows its own size (in terms of pixels)
                 indexNumber,                                                   #A Slither knows it's index number in the slitherfields slitherArray
                 direction = 'R',                                               #Define initial movement direction
                 mnumb=2,                                                       #Define how many members a given slither has upon initialisation
                 basecol = qorange,                                             #Define Slither Fill Color as QtColor
                 linecol = qgrey,                                               #Define Slither Line Color as QtColor
                 engine  = 'player',                                            #Define whether slither controlled by ai or player
                 intraining = False
                 ):
        self.start_x = x                                                        # Starting Position after reset
        self.start_y = y                                                        # Starting Position after reset
        self.x = x
        self.y = y
        self.initial_mnumb = mnumb                                              # Members to Initialize
        self.mnumb = mnumb
        self.initial_direction = direction                                      # Initial direction
        self.lastkey = self.direction = direction
        self.slitherSize = slitherSize
        self.indexNumber=indexNumber
        self.basecolor  = basecol
        self.linecolor  = linecol
        self.controller = engine
        self.intraining = intraining
        self.score = 0
        self.has_eaten = False
        self.is_alive = True
        self.initSlither()

    def returnIndex(self, reset = False):
        return self.indexNumber

    def initSlither(self):
        '''Any Slither consists of a Head and a few Member objects. These are
        initialised within the boundaries of this function.'''

        self.snakeHead = slitherHead(self.x, self.y, self.direction)
        if self.direction == 'R':
            self.memberList=[slitherMember(self.snakeHead,
                            self.x-self.slitherSize, self.y)]
            for i in range(1, self.mnumb):
                self.memberList.insert(0,slitherMember(self.memberList[0],
                                    self.x-self.slitherSize*(1+i), self.y))
        elif self.direction =='L':
            self.memberList=[slitherMember(self.snakeHead,
                            self.x+self.slitherSize, self.y)]
            for i in range(1, self.mnumb):
                self.memberList.insert(0,slitherMember(self.memberList[0],
                                    self.x+self.slitherSize*(1+i), self.y))
        elif self.direction =='U':
            self.memberList=[slitherMember(self.snakeHead,
                            self.x, self.y+self.slitherSize)]
            for i in range(1, self.mnumb):
                self.memberList.insert(0,slitherMember(self.memberList[0],
                                    self.x, self.y+self.slitherSize*(1+i)))
        elif self.direction =='D':
            self.memberList=[slitherMember(self.snakeHead,
                            self.x, self.y-self.slitherSize)]
            for i in range(1, self.mnumb):
                self.memberList.insert(0,slitherMember(self.memberList[0],
                                    self.x, self.y-self.slitherSize*(1+i)))

    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.mnumb = self.initial_mnumb
        self.direction = self.initial_direction
        self.snakeHead.direction = self.initial_direction
        self.snakeHead.x = self.x
        self.snakeHead.y = self.y
        self.is_alive = True
        print(self.memberList[0].Pos)
        del self.memberList[:-self.mnumb]

        print('memberLen  ', len(self.memberList))

        if self.direction =='R':
            for member in self.memberList:
                print(member.Pos)
                member.x = self.x - member.Pos*self.slitherSize
                member.y = self.y


    def returnMemberList(self, c):
        '''This function returns all the Members that have been appended in the
        slither Array'''
        if c == 'a':
            return self.memberList
        elif c <= len(self.memberList):
            return self.memberList[c]


    def memberRange(self):
        return(range(len(self.memberList)))


    def moveHead(self):
        '''Major function that determines the direction in which the Slither
        moves. Every 'turn' the Slither moves the direction of it's size in
        terms of the Slithers own size.'''
        for item in self.memberList:
            item.updateCords()

        if self.direction == 'R':
            self.x +=self.slitherSize
            self.lastkey = 'R'
        elif self.direction == 'U':
            self.y -=self.slitherSize
            self.lastkey = 'U'
        elif self.direction == 'D':
            self.y +=self.slitherSize
            self.lastkey = 'D'
        elif self.direction == 'L':
            self.x -=self.slitherSize
            self.lastkey='L'

        self.snakeHead.updateCords(self.x, self.y, self.lastkey)
        self.has_eaten = False


    def appendMember(self):
        '''This function appends a new Member by initialising a new Member -
        Object at the array's first position'''
        tempx, tempy = self.memberList[0].returnLastPos()
        self.memberList.insert(0, slitherMember(self.memberList[0],
                                                tempx, tempy))
        self.mnumb +=1
        self.score +=1
        self.has_eaten=True

    def return_stats(self, val = 'a'):
        if val == 'all':
            return [self.mnumb, self.color, self.x, self.y, self.color]
        elif val == 'basecolor':
            return self.basecolor
        elif val == 'linecolor':
            return self.linecolor
        elif val == 'score':
            return self.mnumb
        elif val == 'has_eaten':
            return self.has_eaten
        else: print('Invalid Input -> Ignored')


class Food():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def returnCoords(self):
        return self.x, self.y

    def changeCoords(self, x, y):
        self.x = x
        self.y = y


class SlitherField(QtWidgets.QMainWindow):
    def __init__(self,
                 arena_width   = 20,
                 arena_height  = 20,
                 tile_size     = 20,
                 slither_count = 2,
                 food_count    = 25,
                 speed         = 150,
                 slith_att     = [],
                 training_mode = False,
                 episodes      = 100,
                 ):
        super(SlitherField, self).__init__()
        self.viewTable = 'off'                                                  #on or off
        self.setWindowTitle('Slithers')
        self.arenaSize={'width':  arena_width,                                  #Arena Width  - Tiles
                        'height': arena_height}                                 #Arena Height - Tiles
        self.slitherSize = tile_size
        self.setGeometry(50, 50,
                         self.arenaSize['width']*self.slitherSize + 100,
                         self.arenaSize['height']*self.slitherSize + 100)
        self.margins={'upper':50, 'left':50 ,                                   #Treat as constants
                      'right':self.arenaSize['width']*self.slitherSize   +50,
                      'bottom':self.arenaSize['height']*self.slitherSize +50}   #x,y topLeft x,y bottomright
        self.slithCount = slither_count
        self.foodcount  = food_count
        self.emptySpaces = 0
        self.slithAtt    = slith_att
        self.mySlither(self.slithAtt)
        self.init_sNN()
        #self.mySlither(self.slithCount)
        self.height  = abs(self.margins['upper'] - self.margins['bottom'])
        self.length  = abs(self.margins['right'] - self.margins['left'])
        self.speed = speed                                                      #some Timing Measure
        self.training_mode = training_mode
        self.score_label()
        self.timeHandling()
        self.move = 0
        self.total_moves = 0
        self.counter = 0
        self.episodes = episodes
        self.avg_loss = 0
        self.initSlitherField()
        self.foodInit(self.foodcount)
        self.restart_init()
        self.show()
        if self.training_mode:
            self.timer.stop()
            self.train()


    def mySlither(self, atts):
        '''This function initializes all Slither Objects in the Game'''
        l = min(len(atts), self.slithCount)
        self.mySlithers = []
        self.slithAlive = 0

        for s in range(l):
            x,y = self.convCords(atts[s][0], atts[s][1],
                                 type='QT')

            self.mySlithers.append(Slither(
                                           x, y,
                                           self.slitherSize,
                                           indexNumber = s,
                                           mnumb       = atts[s][2],
                                           basecol     = atts[s][3],
                                           linecol     = atts[s][4],
                                           direction   = atts[s][5],
                                           engine      = atts[s][6],
                                           intraining  = atts[s][7]
                                           )
            )

            self.emptySpaces += atts[s][2] + 1
            self.slithAlive +=1


        for s in range(l, self.slithCount):
            print(s)
            x,y = self.convCords(randrange(1, self.arenaSize['width']),
                                randrange(1, self.arenaSize['height']
                                ), type='QT'
                                )
            if x/self.arenaSize['width'] <= 0.5:
                dir='R'
            else:
                dir='L'

            base = QtGui.QColor(randrange(0,255),
                                randrange(0,255),
                                randrange(0,255))

            line = QtGui.QColor(randrange(0,255),
                                randrange(0,255),
                                randrange(0,255))

            self.mySlithers.append(Slither(
                                           x, y,
                                           self.slitherSize,
                                           indexNumber = s,
                                           mnumb       = 2,
                                           basecol     = base,
                                           linecol     = line,
                                           direction   = dir,
                                           engine      = engine.prototype(),
                                           intraining  = False
                                           )
                                    )
            self.emptySpaces += 3
            self.slithAlive +=1

    def score_label(self):
        '''This function initialises and formats the score Label. It is
        called only once in SlitherField.__init__'''
        #SlitherScore 1
        self.mylabel = QtWidgets.QLabel(self)
        self.mylabel.setGeometry(QtCore.QRect(200,10,100,20))
        self.mylabel.setFont(QtGui.QFont('SansSerif',18))
        self.mylabel.setText('0')

        #SlitherScore 2
        self.mylabel2 = QtWidgets.QLabel(self)
        self.mylabel2.setGeometry(QtCore.QRect(1200,10,100,20))
        self.mylabel2.setFont(QtGui.QFont('SansSerif',18))
        self.mylabel2.setText('0')

    def restart_init(self):
        '''This function initialises and formats the Restart button. It is
        called only once in SlitherField.__init__'''
        self.brestart = QtWidgets.QPushButton('Restart', self)
        self.brestart.move(50,25)
        self.brestart.resize(self.brestart.minimumSizeHint())
        self.brestart.clicked.connect(self.restart_Game)

    def restart_Game(self):
        '''This function is connected to the restart button and reinitialises
        all values to enable a restart.'''
        for slither in self.mySlithers:
            slither.reset()

        self.move = 0
        self.emptySpaces = 0
        self.mylabel.setText('0')
        self.slitherField = pd.DataFrame(np.zeros((int(self.height/self.slitherSize+2),
                                      int(self.length/self.slitherSize+2))))

        self.timeHandling()
        #self.timer.start(self.speed)
        print('timer started?')
        self.initSlitherField()
        self.foodInit(self.foodcount)
        self.show()

    def foodInit(self, fnumb):
        self.myfoodList = list()
        tiles = self.arenaSize['width']*self.arenaSize['height']

        for i in range(fnumb):
            tmp = randrange(tiles-self.emptySpaces)
            self.emptySpaces+=1
            self.myfoodList.append(self.randFood(tmp))

    def randFood(self, roll, type = 'init'):
        self.emptySpaces += 1
        tmpy = roll // self.arenaSize['width'] + 1
        tmpx = roll % self.arenaSize['height'] + 1
        while self.slitherField.iloc[tmpy, tmpx] != 0:
            roll += 1
            tmpy = roll // self.arenaSize['width'] + 1
            tmpx = roll %  self.arenaSize['height'] + 1
        self.slitherField.iloc[tmpy, tmpx] = 3
        tmpx, tmpy = self.convCords(tmpx, tmpy, type='QT')
        if type == 'init':                                                      #Called when first initialising Food
            return Food(tmpx, tmpy)
        elif type == 'realloc':                                                 #Called when changing the coordinates of food
            return tmpx, tmpy

    def paintEvent(self, event):
        ''' This Function draws all objects that are displayed within the GuI.
        Note, that all drawing functions must include the painter, initialised
        here and must be included for display.'''
        painter = QtGui.QPainter(self)
        self.drawArena(painter)
        self.drawSlither(painter)
        self.drawFood(painter)

    def drawFood(self, painter):
        ''' This function takes a 'painter' object as an argument, which is
        passed from paintEvent(). This function specifies how to draw the
        Food'''

        self.col = QtGui.QColor(0,0,0)
        painter.setPen(QtGui.QPen(self.col, 0, QtCore.Qt.SolidLine))
        painter.setBrush(QtGui.QBrush(self.col))

        for myfood in self.myfoodList:
            x,y = myfood.returnCoords()
            painter.drawRect(x,y,self.slitherSize, self.slitherSize)

    def drawSlither(self, painter):
        ''' This function takes a 'painter' object as an argument, which is
        passed from paintEvent(). This function specifies how to draw the
        slither'''
        for s in range(len(self.mySlithers)):
            if self.mySlithers[s].is_alive:
                basecol = self.mySlithers[s].return_stats(val = 'basecolor')
                linecol = self.mySlithers[s].return_stats(val = 'linecolor')
                painter.setPen(QtGui.QPen(basecol, 0, QtCore.Qt.SolidLine))
                painter.setBrush(QtGui.QBrush(linecol))
                painter.drawEllipse(self.mySlithers[s].returnPosx(),
                self.mySlithers[s].returnPosy(), self.slitherSize, self.slitherSize)
                for m in self.mySlithers[s].memberRange():
                    painter.drawRect(self.mySlithers[s].memberList[m].returnPosx(), #We are accessing the Memberlist containing
                                     self.mySlithers[s].memberList[m].returnPosy(), #the Member Objects 'm' of Slither 's'
                                     self.slitherSize, self.slitherSize)

    def drawArena(self, painter):
        '''This function is used to draw the SlitherField Arena, by using lines.
        Note, that using lines, as opposed to a rectangle is more efficient,
        since we do not need to cope with which layer is displayed at present.
        '''

        painter.setPen(QtGui.QPen(qgrey, 3, QtCore.Qt.SolidLine))

        painter.drawLine(self.margins['upper'], self.margins['left'],           #Left Line
                         self.margins['left'], self.margins['bottom'])
        painter.drawLine(self.margins['upper'], self.margins['bottom'],         #Right Line
                         self.margins['right'], self.margins['bottom'])

        painter.drawLine(self.margins['upper'], self.margins['left'],
                         self.margins['right'], self.margins['left'])
        painter.drawLine(self.margins['right'], self.margins['left'],
                         self.margins['right'], self.margins['bottom'])

    def timeHandling(self):
        '''This function is called at the init and initialises a timer, that
        calls the moveSlither function after every tick (self.speed), which is
        specified in the SlitherField.__init__'''
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.moveSlither)
        self.timer.start(self.speed)

    def moveSlither(self):
        '''This function calls all UDF-Function required to moving a Slither,
        that is (i) moving the SlitherClass, (ii) updating the slitherField -
        Matrix and (iii) repainting'''
        for slither in self.mySlithers:
            if slither.is_alive and slither.controller != 'player' and slither.intraining:
                tmp = slither.returnCords()
                tx, ty = self.convCords(tmp[1],tmp[0])
                dir = slither.\
                        controller.set_action(self.slitherField,
                                              [tx, ty],
                                              self.makeFoodList())

                self.engine_move(slither.returnIndex(), dir)

        for slither in self.mySlithers:
            if slither.is_alive:
                slither.moveHead()

        self.updateSlitherinField()
        self.repaint()

    def engine_move(self, sindex, d):
        if d == 0:
            if self.mySlithers[sindex].is_alive:
                self.mySlithers[sindex].turnUp()
        elif d == 1:
            if self.mySlithers[sindex].is_alive:
                self.mySlithers[sindex].turnDown()
        elif d == 2:
            if self.mySlithers[sindex].is_alive:
                self.mySlithers[sindex].turnLeft()
        elif d == 3:
            if self.mySlithers[sindex].is_alive:
                self.mySlithers[sindex].turnRight()

    def keyPressEvent(self, e):
        ''' This function catches all Key Inputs and relays them to the
        respective Slithers'''

        if e.key() == QtCore.Qt.Key_W:
            if self.mySlithers[0].is_alive:
                self.mySlithers[0].turnUp()
        elif e.key() == QtCore.Qt.Key_S:
            if self.mySlithers[0].is_alive:
                self.mySlithers[0].turnDown()
        elif e.key() == QtCore.Qt.Key_A:
            if self.mySlithers[0].is_alive:
                self.mySlithers[0].turnLeft()
        elif e.key() == QtCore.Qt.Key_D:
            if self.mySlithers[0].is_alive:
                self.mySlithers[0].turnRight()
        elif e.key() == QtCore.Qt.Key_I:
            if self.mySlithers[1].is_alive:
                self.mySlithers[1].turnUp()
        elif e.key() == QtCore.Qt.Key_K:
            if self.mySlithers[1].is_alive:
                self.mySlithers[1].turnDown()
        elif e.key() == QtCore.Qt.Key_J:
            if self.mySlithers[1].is_alive:
                self.mySlithers[1].turnLeft()
        elif e.key() == QtCore.Qt.Key_L:
            if self.mySlithers[1].is_alive:
                self.mySlithers[1].turnRight()

    def initSlitherField(self):
        '''This function initialises the SlitherField, in particular the death
        boundaries. However, it does not initiliase the Food or the Slithers.
        See slitherInField and foodInit.'''

        self.slitherField = pd.DataFrame(np.zeros((int(self.height/self.slitherSize+2),
                                      int(self.length/self.slitherSize+2))))

        fieldheight, fieldwidth = self.slitherField.shape

        #Setting the 'Bounds of the SlitherField
        self.slitherField.iloc[[0, fieldheight-1],:] = 1
        self.slitherField.iloc[:,[0,fieldwidth-1]] = 1
        self.slitherInField()                                                   #Updates all the Slither Coords in the SlitherField

        #Create Viewer
        if self.viewTable == 'on':
            self.view = QtWidgets.QTableView()
            self.model = myDataTable.PandasModel(self.slitherField)
            self.view.setModel(self.model)
            self.view.show()

    def slitherInField(self):
        '''This function updates the Slithers Position into the SlitherField, it
        must be called only once within the SlitherField Initialisation, as Sli-
        ther Movement is handeled by updateSlitherinField, which is a lotfaster
        '''
        for s in range(len(self.mySlithers)):
#        if self.mySlithers[s].is_alive:
            tempx, tempy = self.mySlithers[s].returnCords()
            tempx, tempy = self.convCords(tempx, tempy, type='SF')
            if self.slitherField.iloc[tempy, tempx] != 1:
                self.slitherField.iloc[tempy, tempx] = self.mySlithers[s].indexNumber+4
                for m in self.mySlithers[s].memberRange():
                    tempx, tempy = self.mySlithers[s].memberList[m].returnCords()
                    tempx, tempy = self.convCords(tempx, tempy)
                    self.slitherField.iloc[tempy, tempx] = 2

    def updateSlitherinField(self):
        '''This Function Updates the SlitherField Matrix, assuming that 1
        Move Slither was called. The new Head Position gets updated to 'H',
        the first Member has to have the Position 2, hence the 'H' gets over-
        written to 2. Finally, the last Members previous Position is set to
        '0' as the last Member has moved 1 space.'''

        #First Member must now be 2 instead of 'H'
        for slither in self.mySlithers:
            if slither.is_alive:
                tempx, tempy = slither.\
                    memberList[len(slither.memberList)-1].returnCords()         #The slithers 2nd highest Member, is
                tempx, tempy = self.convCords(tempx, tempy)                     #the one where the 'H' must be overwritten to 2
                self.slitherField.iloc[tempy, tempx] = 2

                #Last Member must now be '0' instead of 'H'
                tempx, tempy = slither.memberList[0].returnLastPos()
                if tempx:                                                       #Hypothetically tempx does not exist,
                    tempx, tempy = self.convCords(tempx, tempy)                 #whenevever we are looking at the first frame
                    self.slitherField.iloc[tempy, tempx] = float(0)

        for slither in self.mySlithers:
            if slither.is_alive:
                #Update Head Position
                tempx, tempy = slither.returnCords()                            #!Take Care, by construction,
                tempx, tempy = self.convCords(tempx, tempy)                     #the call Slither.returnCords, returns the Heads Position
                #before overwriting, check if the head leads to the termination
                self.slitherStatus(tempx, tempy, slither.indexNumber)
                if self.slitherField.iloc[tempy, tempx] != 1:
                    self.slitherField.iloc[tempy, tempx] = slither.indexNumber + 4

        #Viewer
        if self.viewTable == 'on':
            self.view.setModel(self.model)
            self.view.show()

    def convCords(self, x, y, type='SF'):
        '''Within this script two types of Coordinates are being used:
        1) Pixel Coordinates used by PyQt5 and
        2) SlitherField Coordinates
        This function convertes Pixel Coordinates into SliterField Coordinates,
        that are used to determine the Position within the SlitherField-Matrix
        '''
        if type =='SF':
            _tempx = x/self.slitherSize+1
            _tempx = int(_tempx - self.margins['left']/self.slitherSize)
            _tempy = y/self.slitherSize + 1
            _tempy = _tempy - self.margins['upper']/self.slitherSize
            return int(_tempx), int(_tempy)

        if type == 'QT':
            #The first valid field is (margins_left, margins_upper)
            _tempx = x * self.slitherSize + \
                self.margins['left'] - self.slitherSize
            _tempy = y * self.slitherSize + \
                self.margins['upper'] - self.slitherSize
            return int(_tempx), int(_tempy)

    def slitherStatus(self, tempx, tempy, sindex):
        ''' Arguments: SlitherField Coordinates x, y. This function return
        the status depending on the Slitherfield Coordinates, assuming that the
        Slither Heads coordinates were passed. Possible Outcomes: The head is on
        a food Field, on a Death Field in (D = Border and M = Member) '''
        if self.slitherField.iloc[tempy, tempx] == 3:
            self.mySlithers[sindex].appendMember()
            self.printScore()
            tempi = self.foodPosition(tempx, tempy)
            self.placenewFood(tempi)
        elif self.slitherField.iloc[tempy, tempx] == 1:
            self.slither_death(sindex)
        elif self.slitherField.iloc[tempy, tempx] == 2:
            self.slither_death(sindex)
        elif self.slitherField.iloc[tempy, tempx] in range(4,20):               #Maximum Number of SLither allowed
            # If we have a collision with the head, the longer snake survives,
            # or both die
            tempx, tempy = self.mySlithers[sindex].returnCords()
            collision_list = {                                                  #Dictionary with slitherindex and score
                sindex:self.mySlithers[sindex].return_stats('score')
                }
            for s in range(len(self.mySlithers)):
                if self.mySlithers[s].is_alive and s != sindex \
                    and [tempx, tempy] == list(self.mySlithers[s].returnCords()):
                    collision_list.update(
                      {s : self.mySlithers[s].return_stats('score')}
                    )

            collision_max = max(collision_list.values())
            max_counter = 0                                                     #Kill all snakes if max is in list >1
            for k, v in collision_list.items():
                if v == collision_max:
                    max_counter += 1
            if max_counter >=2:                                                 #If 2+ snakes have max size, all snakes will die
                collision_max += 1
            for k, v in collision_list.items():
                if v < collision_max:
                    print('max: {}, killing {}'.format(collision_max, k))
                    self.slither_death(k)
        if self.slithAlive < 1:
            self.timer.stop()
            if self.training_mode and self.counter < self.episodes:
                self.counter += 1
                self.restart_Game()



    def slither_death(self, sindex):
        print('Slither {} died'.format(sindex))
        self.arg_score = -100

        if self.training_mode:
            self.sE[0].memory.add_sample(
                (
                self.previous_state,
                self.action,
                self.arg_score,                                                 # Penalty for dying
                self.slitherField.copy()
                )
            )

        for m in self.mySlithers[sindex].memberList:
            tempx, tempy = m.returnCords()
            tempx, tempy  = self.convCords(tempx, tempy)
            self.slitherField.iloc[tempy, tempx]=0
        tempx, tempy = self.mySlithers[sindex].returnCords()
        tempx, tempy = self.convCords(tempx, tempy)
        if self.slitherField.iloc[tempy, tempx]!=1:
            self.slitherField.iloc[tempy, tempx]=0
        self.mySlithers[sindex].is_alive = False
        self.slithAlive -=1
        self.total_moves += self.move

        if self.training_mode:
            print(f"Episode: {self.counter}, Score: {self.mySlithers[0].score}, 'Moves': {self.move}, avg loss: {self.avg_loss/self.move:.3f}, eps: {self.eps:.3f}")




    def printScore(self):
        ''' This function prints the Score of the Slither in question.
        We still need to update '''
        if self.mySlithers[0].is_alive:
            self.myScore = self.mySlithers[0].memberList[0].returnPosition()-2
            self.mylabel.setText(str(self.myScore))
        if len(self.mySlithers)>1 and self.mySlithers[1].is_alive:               #If a 2nd Slither exists,
            self.myScore2 = self.mySlithers[1].memberList[0].returnPosition()-2 #it's score is set to 0
            self.mylabel2.setText(str(self.myScore2))

    def placenewFood(self, i):
        ''' This function takes the index of a Food in the Food-List and
        randomly changes it's coordinates - which is necessary whenevever a
        Snake eats Food'''
        tiles = self.arenaSize['width']*self.arenaSize['height']
        tmp = randrange(tiles-self.emptySpaces)
        self.emptySpaces += 1

        tmpx, tmpy = self.randFood(tmp, type='realloc')
        self.myfoodList[i].changeCoords(tmpx, tmpy)

    def foodPosition(self, x, y):
        '''The food is initialised in a list. This function takes x, y
        slitherField Coordinates as Inputs and returns the Foods Position in
        the Food-List'''
        for i in range(len(self.myfoodList)):
            tempx , tempy = self.myfoodList[i].returnCoords()
            tempx, tempy = self.convCords(tempx, tempy)

            if ((tempx == x) & (tempy == y) ):
                return i

    def getEmptySpaces(self):
        '''Returns empty spaces in the slitherField'''
        emptyspaces = 0
        for i in range(1, self.arenaSize['width']+1):
            for j in range(1, self.arenaSize['height']+1):
                if self.slitherField.iloc[i,j] == 0:
                    emptyspaces +=1
        return emptyspaces

    def makeFoodList(self):
        foodList = []
        for item in self.myfoodList:
            tmpx, tmpy = self.convCords(item.returnCoords()[0],
                                        item.returnCoords()[1])
            foodList.append([tmpy, tmpx])
        return foodList

    def init_sNN(self):
        self.sE = []
        for s in self.mySlithers:
            if s.intraining:
                self.sE.append(sai.sNN(s.indexNumber))

        self.eps = self.sE[0].MAX_EPSILON
        self.arg_score = 0

    def train_iteration(self):
        self.previous_state = self.slitherField.copy()
        self.action = self.sE[0].choose_action(
            self.slitherField,
            self.sE[0].primary_network,
            self.eps
            )
        self.move += 1
        self.engine_move(0, self.action)
        self.moveSlither()


        if self.mySlithers[0].has_eaten:
            self.arg_score = self.mySlithers[0].has_eaten*1
        else:
            self.arg_score = 1



        self.sE[0].memory.add_sample(
            (
            self.previous_state,
            self.action,
            self.arg_score,
            self.slitherField.copy()
            )
        )


        self.direction = {
          0: 'up',
          1: 'down',
          2: 'left',
          3: 'right'
        }

        loss = self.sE[0].train(self.sE[0].primary_network,
                                self.sE[0].memory,
                                # self.sE[0].target_network
                                )

        self.avg_loss += loss
        self.eps = self.sE[0].MIN_EPSILON + (self.sE[0].MAX_EPSILON - self.sE[0].MIN_EPSILON) * math.exp(- self.sE[0].LAMBDA * self.total_moves)

        if self.slithAlive == 0:
            self.avg_loss /= self.move
            print(f"Episode: {self.counter}, Reward: {self.mySlithers[0].score}, 'Moves': {self.move}, avg loss: {self.avg_loss:.3f}, eps: {self.eps:.3f}")
            with self.sE[0].train_writer.as_default():
                tf.summary.scalar('reward', self.move, step=self.counter)
                tf.summary.scalar('avg loss', self.avg_loss, step=self.counter)
            self.timer.stop()

    def train(self):
        render = False
        double_q = True
        steps = 0
        self.timer.timeout.connect(self.train_iteration)
        self.timer.start()


if __name__=='__main__':
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    else:
        print('QApplication instance already exists: %s' % str(app))

    s1 = [5,5,
          2,
          qorange,
          qgrey,
          'R',
          'player',
          True]

    s2 = [6,12,
          2,
          qred,
          qdark,
          'R',
          'player',
          False]
          # engine.prototype()]

    slithers=[s1,s2]

    snake = SlitherField(slither_count = 1,
                         slith_att     = slithers,
                         episodes      = 1,
                         training_mode = False,
                         arena_width   = 10,
                         arena_height  = 10,
                         food_count    = 10)
    app.exec_()
