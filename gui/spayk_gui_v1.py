from PySide6.QtCore import Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QColor    
from PySide6.QtCore import Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QApplication
import sys

class SpaykMainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("SPAYK :: v1.0")

        # Menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")
        self.model_menu = self.menu.addMenu("Models")
        self.run_menu = self.menu.addMenu("Simulation")

        ## Exit QAction
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)

        self.file_menu.addAction(exit_action)

        # Status Bar
        self.status = self.statusBar()
        self.status.showMessage("SPAYK GUI has been initialized successfully.")

        # Window dimensions
        geometry = self.screen().availableGeometry()
      

 


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = SpaykMainWindow()

    window.show()

    sys.exit(app.exec())
