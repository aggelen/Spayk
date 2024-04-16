import sys
import os
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtUiTools import QUiLoader

CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
UI_PATH = os.path.join(CURRENT_DIRECTORY, "ui/test1.ui")


loader = QUiLoader()
app = QtWidgets.QApplication(sys.argv)
window = loader.load(UI_PATH, None)
window.show()
app.exec_()