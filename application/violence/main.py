import os
import sys
import signal

from PyQt5 import QtWidgets
from main_app.controller.main_controller import MainController

os.environ["DISPLAY"] = ":0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtWidgets.QApplication(sys.argv)
    main_controller = MainController()
    sys.exit(app.exec_())
