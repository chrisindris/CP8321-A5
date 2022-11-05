import sys
import time

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class DlgMain(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My GUI")

        self.prg = QProgressBar()
        self.prg.setStyle(QStyleFactory.create("Windows"))
        self.prg.setTextVisible(True)

        self.btnStart = QPushButton("Start")
        self.btnStart.clicked.connect(self.evt_btnStart_clicked)

        self.dial = QSlider()
        self.lcd = QLCDNumber()
        self.dial.valueChanged.connect(self.lcd.display)

        self.lytLCD = QHBoxLayout()
        self.lytLCD.addWidget(self.dial)
        self.lytLCD.addWidget(self.lcd)

        self.lytMain = QVBoxLayout()
        self.lytMain.addWidget(self.prg)
        self.lytMain.addWidget(self.btnStart)
        self.lytMain.addLayout(self.lytLCD)
        self.setLayout(self.lytMain)

    def evt_btnStart_clicked(self):
        self.worker = WorkerThread()
        self.worker.start()
        self.worker.worker_complete.connect(self.evt_worker_finished)
        self.worker.update_progress.connect(self.evt_update_progress)

    def evt_worker_finished(self, emp):
        msg = QMessageBox()
        msg.setWindowTitle("Tutorial on PyQt5")
        msg.setText("{} {} {}".format(emp["emp_id"], emp["fn"], emp["ln"]))
        msg.setIcon(QMessageBox.Information)

        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ignore)
        msg.setInformativeText("Informative Text")
        msg.setDetailedText("details...")
        msg.exec_()

    def evt_update_progress(self, val):
        self.prg.setValue(val)


class WorkerThread(QThread):

    update_progress = pyqtSignal(int)
    worker_complete = pyqtSignal(dict)

    def run(self):
        for x in range(0, 110, 10):
            print(x)
            time.sleep(0.5)
            self.update_progress.emit(x)
        self.worker_complete.emit(
            {"emp_id": 1234, "fn": "Chris", "ln": "Indris"})


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlgMain = DlgMain()
    dlgMain.show()
    sys.exit(app.exec_())
