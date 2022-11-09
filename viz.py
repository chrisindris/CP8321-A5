# -*- coding: utf-8 -*-
""" Visualization Code
This script creates and manages the GUI. It receives weights from regular
intervals, generates images using matplotlib and then embeds them into the GUI.

Built with PyQt5 UI Code Generator (Designer) 5.15.7
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import pyplot as plt
import numpy as np

import mnist_cnn_torch


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        """ Build GUI and create instance of WorkerThread (mnist_cnn_torch)
        for backpropagation calculation.

        :param MainWindow object.
        :return: none.
        """
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 1400)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Titles and Menu

        self.label_title = self.paint_label(10, -40, 341, 131,
                                            "label_title", 27)

        self.label_weights = self.paint_label(485, -40, 341, 131,
                                            "label_weights", 20)

        self.label_architecture = self.paint_label(850, -40, 341, 131,
                                              "label_architecture", 20)


        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(0, 100, 201, 61))
        self.pushButton.setObjectName("pushButton")

        # Photos (Dynamic) + Labels

        self.conv1_photo = \
            self.paint_image(370, 100, 300, 255, "img4.png", "conv1_photo")
        self.label_conv1_photo = self.paint_label(410, 0, 300, 255,
                                                   "label_conv1_photo", 12)

        self.conv2_photo = \
            self.paint_image(370, 350, 300, 255, "img3.png", "conv2_photo")
        self.label_conv2_photo = self.paint_label(410, 295, 300, 255,
                                                  "label_conv2_photo", 12)

        self.lin1_photo = \
            self.paint_image(210, 420, 600, 800, "img.png", "lin1_photo")
        self.label_lin1_photo = self.paint_label(290, 630, 300, 255,
                                                  "label_lin1_photo", 12)

        self.lin2_photo = \
            self.paint_image(210, 750, 600, 510, "img2.png", "lin2_photo")
        self.label_lin2_photo = self.paint_label(285, 820, 300, 255,
                                                 "label_lin2_photo", 12)

        # Photos (Static)

        self.architecture_photo = \
            self.paint_image(800, 80, round(286 * 0.8), round(1436 * 0.8),
                             "./files/architecture.png",
                             "architecture_photo")

        # Functionality

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.evt_btnStart_clicked)

        self.worker = mnist_cnn_torch.WorkerThread()

    def evt_btnStart_clicked(self):
        """ Button event handler. Starts the training (backpropagation)
        :return: none.
        """
        self.worker.start()
        self.worker.update_progress.connect(self.evt_update_progress)

    def evt_update_progress(self, val):
        """ Builds and sets the dynamic GUI images.
        :param val: Array of current weight matrices.
        :return:
        """
        plt.figure(figsize=(20, 17), frameon=False)  # 20, 17
        plt.imshow(val[3].weight.detach().numpy())
        plt.axis('off')
        plt.savefig("img.png")
        self.lin1_photo.setPixmap(QtGui.QPixmap("img.png"))

        plt.figure(figsize=(20, 17), frameon=False)
        plt.imshow(val[4].weight.detach().numpy())
        plt.axis('off')
        plt.savefig("img2.png")
        self.lin2_photo.setPixmap(QtGui.QPixmap("img2.png"))

        plt.figure(figsize=(20, 17), frameon=False)
        plt.axis('off')
        A = np.zeros((10, 25))
        k = 0
        for i in range(2):
            for j in range(5):
                A[5 * i:5 * (i + 1), 5 * j:5 * (j + 1)] = val[0].weight.detach().numpy()[k][0]
                k += 1
        plt.imshow(A)
        plt.savefig("img3.png")
        self.conv2_photo.setPixmap(QtGui.QPixmap("img3.png"))

        plt.figure(figsize=(20, 17), frameon=False)
        plt.axis('off')
        A = np.zeros((20, 25))
        k = 0
        for i in range(4):
            for j in range(5):
                A[5 * i:5 * (i + 1), 5 * j:5 * (j + 1)] = \
                val[1].weight.detach().numpy()[k][0]
                k += 1
        plt.imshow(A)
        plt.savefig("img4.png")
        self.conv1_photo.setPixmap(QtGui.QPixmap("img4.png"))

        plt.close('all')

    # --- GUI Building Helpers ---

    def retranslateUi(self, MainWindow):
        """ Sets the label text
        :param MainWindow:
        :return:
        """
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CP8321 A5"))

        self.label_title.setText(_translate("MainWindow", "MNIST CNN in Torch"))
        self.label_weights.setText(_translate("MainWindow", "Weights"))
        self.label_architecture.setText(_translate("MainWindow", "Architecture"))

        self.pushButton.setText(_translate("MainWindow", "Train"))

        self.label_conv1_photo.setText(_translate("MainWindow", "First Conv Layer: 20 5x5 filters"))
        self.label_conv2_photo.setText(_translate("MainWindow", "Second Conv Layer: 10 5x5 filters"))
        self.label_lin1_photo.setText(_translate("MainWindow", "First Linear Layer: 50x320 weights"))
        self.label_lin2_photo.setText(_translate("MainWindow", "Second Linear Layer: 10x50 weights"))

    def paint_image(self, x, y, w, h, default, name):
        img = QtWidgets.QLabel(self.centralwidget)
        img.setGeometry(QtCore.QRect(x, y, w, h))
        img.setText("")
        img.setPixmap(QtGui.QPixmap(default))
        img.setScaledContents(True)
        img.setObjectName(name)
        return img

    def paint_label(self, x, y, w, h, name, font_size):
        lab = QtWidgets.QLabel(self.centralwidget)
        lab.setGeometry(QtCore.QRect(x, y, w, h))
        lab.setObjectName(name)
        font_title = QtGui.QFont()
        font_title.setPointSize(font_size)
        lab.setFont(font_title)
        return lab


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
