# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:16:05 2019

@author: DE104760
"""

#Standard Libraries
import pandas as pd
import sys

#Third Party Libraries
from PyQt5 import QtGui, QtWidgets, QtCore

class PandasModel(QtCore.QAbstractTableModel):
    '''
    This class defines a Model in terms of a pd Dataframe using predefined 
    functions in QT
    '''
    
    def __init__(self, df = pd.DataFrame(), parent=None): 
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df
        

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()
        elif orientation == QtCore.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if not index.isValid():
            return QtCore.QVariant()

        return QtCore.QVariant(str(self._df.ix[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True


    def rowCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.columns)

# =============================================================================
#     def sort(self, column, order):
#         colname = self._df.columns.tolist()[column]
#         self.layoutAboutToBeChanged.emit()
#         self._df.sort_values(colname, ascending= order == QtCore.Qt.AscendingOrder, inplace=True)
#         self._df.reset_index(inplace=True, drop=True)
#         self.layoutChanged.emit()
# =============================================================================
        
if __name__=="__main__":
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    else:
        print('QApplication instance already exists: %s' % str(app))
    
    
    
    #mypath = r"C:\Users\DE104760\Desktop\FSITToolbox\Interface\IT_Reports\testdata.xlsx"
    mypath = r"C:\Users\DE104760\Desktop\FSITToolbox\Interface\Data.xlsx"
    mydf = pd.read_excel(mypath)#, sheet_name = 'Sheet1')
    
    view = QtWidgets.QTableView()
    model = PandasModel(mydf)
        
    view.setModel(model)
    view.show()
    app.exec()
