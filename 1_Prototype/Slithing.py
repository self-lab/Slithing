'''
Created on Sun May 26 16:49:51 2019
@author: Milan Reichmann
'''

#Standard Libraries
import sys                                                                      #Systemfuncitons, such as exit Programm
from random import randrange                                                    #Random N. Generator, i.e. Food Placement
import numpy as np
import pandas as pd
from math import floor

#Third Party Libraries - Professional
from PyQt5 import QtGui, QtWidgets, QtCore                                      #API für GuI, Widgets

#Third Party Libraries - Amateur
import TableView as myDataTable

#User defined


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
    def __init__(self, x, y,
                 slitherSize, indexNumber,
                 direction = 'R', mnumb=2, basecol = qorange, linecol = qgrey   #Passing a Qt Color to col
                 ):
        self.x = x
        self.y = y
        self.mnumb = mnumb
        self.direction = direction
        self.slitherSize = slitherSize
        self.indexNumber=indexNumber
        self.basecolor  = basecol
        self.linecolor  = linecol
        self.initSlither()

    def returnIndex(self):
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


    def appendMember(self):
        '''This function appends a new Member by initialising a new Member -
        Object at the array's first position'''
        tempx, tempy = self.memberList[0].returnLastPos()
        self.memberList.insert(0, slitherMember(self.memberList[0],
                                                tempx, tempy))
        self.mnumb +=1

    def return_stats(self, val = 'a'):
        if val == 'all':
            return [self.mnumb, self.color, self.x, self.y, self.color]
        elif val == 'basecolor':
            return self.basecolor
        elif val == 'linecolor':
            return self.linecolor
        elif val == 'score':
            return self.mnumb
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
    def __init__(self):
        super(SlitherField, self).__init__()
        self.viewTable = 'off'                                                  #on or off
        self.setWindowTitle('Slithers')
        self.arenaSize={'width':  20,  #Arena Width  - Tiles
                        'height': 20}  #Arena Height - Tiles

        self.slitherSize = 10
        self.setGeometry(50, 50, self.arenaSize['width']*self.slitherSize + 100,
                         self.arenaSize['height']*self.slitherSize + 100)
        self.margins={'upper':50, 'left':50 ,                                   #Treat as constants
                      'right':self.arenaSize['width']*self.slitherSize   +50,
                      'bottom':self.arenaSize['height']*self.slitherSize +50}   #x,y topLeft x,y bottomright
        self.slithCount = 2
        self.emptySpaces = 0
        self.mySlither(self.slithCount)
        self.lastKey = 'R'
        self.height  = abs(self.margins['upper'] - self.margins['bottom'])
        self.length  = abs(self.margins['right'] - self.margins['left'])
        self.speed = 150                                                          #some Timing Measure
        self.timeHandling()
        self.gameStatus = 'Alive'
        self.initSlitherField()
        self.foodInit()
        self.score_label()
        self.restart_init()
        self.show()

    def mySlither(self, nslith):
        '''This function initialises all the Slither-Objects in the Field'''
        self.mySlithers = list()

        _tempx, _tempy = self.convCords(2,2, type='QT')

        self.mySlithers.append(Slither(_tempx,
                                       _tempy,
                                       self.slitherSize,
                                       indexNumber=0,
                                       basecol=qorange,
                                       linecol=qgrey,
                                       direction = 'R'
                                       )
                                )

        if nslith >= 2:
            _tempx, _tempy = self.convCords(10,2, type='QT')
            self.mySlithers.append(Slither(_tempx,
                                           _tempy,
                                           self.slitherSize,
                                           indexNumber=1,
                                           basecol=qred,
                                           linecol=qdark,
                                           direction = 'L')
                                  )

        self.emptySpaces += nslith*3

        self.slithAlive = 0
        for s in self.mySlithers:
            if s != 'dead':
                self.slithAlive +=1

        print(self.slithAlive)
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
            del slither

        self.emptySpaces = 0
        self.mylabel.setText('0')
        self.slitherField = pd.DataFrame(np.zeros((int(self.height/self.slitherSize+2),
                                      int(self.length/self.slitherSize+2))))
        self.mySlither(self.slithCount)
        self.lastKey = 'R'
        self.timeHandling()
        self.gameStatus = 'Alive'
        self.initSlitherField()
        self.foodInit()
        self.show()

    def foodInit(self, fnumb = 25):
        '''TO DO MOVE PART OF THIS SECTION DOWN TO RANDFOOD AND FINISH THE FOOD CLASS REPLACEMENT'''

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
        while self.slitherField.iloc[tmpy, tmpx]!=0:
            roll += 1
            tmpy = roll // self.arenaSize['width'] + 1
            tmpx = roll %  self.arenaSize['height'] + 1
            print('Debugging: NonTrivialCase')
        self.slitherField.iloc[tmpy, tmpx] = 'F'
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
            if self.mySlithers[s] != 'dead':
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
            if slither != 'dead':
                slither.moveHead()
        self.updateSlitherinField()
        self.repaint()


    # def keyPressEvent(self, e):
    #     ''' This function catches all Key Inputs and relays them to the
    #     respective Slithers'''
    #     if e.key() == QtCore.Qt.Key_W:
    #         self.mySlithers[0].turnUp()
    #     elif e.key() == QtCore.Qt.Key_S:
    #         self.mySlithers[0].turnDown()
    #     elif e.key() == QtCore.Qt.Key_A:
    #         self.mySlithers[0].turnLeft()
    #     elif e.key() == QtCore.Qt.Key_D:
    #         self.mySlithers[0].turnRight()
    #     elif e.key() == QtCore.Qt.Key_I:
    #         self.mySlithers[1].turnUp()
    #     elif e.key() == QtCore.Qt.Key_K:
    #         self.mySlithers[1].turnDown()
    #     elif e.key() == QtCore.Qt.Key_J:
    #         self.mySlithers[1].turnLeft()
    #     elif e.key() == QtCore.Qt.Key_L:
    #         self.mySlithers[1].turnRight()


    def initSlitherField(self):
        '''This function initialises the SlitherField, in particular the death
        boundaries. However, it does not initiliase the Food or the Slithers.
        See slitherInField and foodInit.'''

        self.slitherField = pd.DataFrame(np.zeros((int(self.height/self.slitherSize+2),
                                      int(self.length/self.slitherSize+2))))

        fieldheight, fieldwidth = self.slitherField.shape

        #Setting the 'Bounds of the SlitherField
        self.slitherField.iloc[[0, fieldheight-1],:] = 'D'
        self.slitherField.iloc[:,[0,fieldwidth-1]] = 'D'
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
            tempx, tempy = self.mySlithers[s].returnCords()
            tempx, tempy = self.convCords(tempx, tempy)
            self.slitherField.iloc[tempy, tempx] = 'H'
            for m in self.mySlithers[s].memberRange():
                tempx, tempy = self.mySlithers[s].memberList[m].returnCords()
                tempx, tempy = self.convCords(tempx, tempy)
                self.slitherField.iloc[tempy, tempx] = 'M'


    def updateSlitherinField(self):
        '''This Function Updates the SlitherField Matrix, assuming that 1
        Move Slither was called. The new Head Position gets updated to 'H',
        the first Member has to have the Position 'M', hence the 'H' gets over-
        written to 'M'. Finally, the last Members previous Position is set to
        '0' as the last Member has moved 1 space.'''

        #First Member must now be 'M' instead of 'H'
        for slither in self.mySlithers:
            if slither != 'dead':
                tempx, tempy = slither.\
                    memberList[len(slither.memberList)-1].returnCords()             #The slithers 2nd highest Member, is
                tempx, tempy = self.convCords(tempx, tempy)                         #the one where the 'H' must be overwritten to 'M'
                self.slitherField.iloc[tempy, tempx] = 'M'

                #Last Member must now be '0' instead of 'H'
                tempx, tempy = slither.memberList[0].returnLastPos()
                if tempx:                                                           #Hypothetically tempx does not exist,
                    tempx, tempy = self.convCords(tempx, tempy)                     #whenevever we are looking at the first frame
                    self.slitherField.iloc[tempy, tempx] = float(0)

        for slither in self.mySlithers:
            if slither != 'dead':
                #Update Head Position
                tempx, tempy = slither.returnCords()                                #!Take Care, by construction,
                tempx, tempy = self.convCords(tempx, tempy)                         #the call Slither.returnCords, returns the Heads Position
                #before overwriting, check if the head leads to the termination
                self.slitherStatus(tempx, tempy, slither.indexNumber)
                self.slitherField.iloc[tempy, tempx] = 'H'

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
        if self.slitherField.iloc[tempy, tempx] == 'F':
            self.mySlithers[sindex].appendMember()
            self.printScore()
            #print(tempy, tempx)
            tempi = self.foodPosition(tempx, tempy)
            self.placenewFood(tempi)
        elif self.slitherField.iloc[tempy, tempx] == 'D':
            self.slither_death(sindex)
        elif self.slitherField.iloc[tempy, tempx] == 'M':
            self.timer.stop()
        elif self.slitherField.iloc[tempy, tempx] == 'H':
            # If we have a collision with the head, the longer snake survives,
            # or both die
            tempx, tempy = self.mySlithers[sindex].returnCords()
            collision_list = {                                                  #Dictionary with slitherindex and score
                sindex:self.mySlithers[sindex].return_stats('score')
                }
            for s in range(len(self.mySlithers)):
                if self.mySlithers[s] != 'dead' and s != sindex \
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


            print(collision_list)
            # print('sindex: {} and other slither: {}'.format(sindex, s))
            # a = self.mySlithers[s].return_stats('score')
            # b = self.mySlithers[sindex].return_stats('score')
            # print('Score sindex: {}, Score other: {}'.format(b,a))


        if self.slithAlive < 1:
            self.timer.stop()


    def slither_death(self, sindex):
        print('Slither {} died'.format(sindex))
        for m in self.mySlithers[sindex].memberList:
            tempx, tempy = m.returnCords()
            tempx, tempy  = self.convCords(tempx, tempy)
            self.slitherField.iloc[tempy, tempx]=0
        tempx, tempy = self.mySlithers[sindex].returnCords()
        tempx, tempy = self.convCords(tempx, tempy)
        if self.slitherField.iloc[tempy, tempx]!='D':
            self.slitherField.iloc[tempy, tempx]=0
        print(self.slitherField)
        self.slithAlive -=1
        self.mySlithers[sindex]='dead'

    def printScore(self):
        ''' This function prints the Score of the Slither in question.
        We still need to update '''
        if self.mySlithers[0] != 'dead':
            self.myScore = self.mySlithers[0].memberList[0].returnPosition()-2
            self.mylabel.setText(str(self.myScore))
        if len(self.mySlithers)>1 and self.mySlithers[1]!='dead':                                                  #If a 2nd Slither exists,
            self.myScore2 = self.mySlithers[1].memberList[0].returnPosition()-2     #it's score is set to 0
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



'''Note, that the if clause below is relevant for a few IDE's such as Spyder
only - Thank God I switched to Atom'''
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
else:
    print('QApplication instance already exists: %s' % str(app))

snake = SlitherField()
app.exec_()
