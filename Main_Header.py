import sys
import subprocess
import numpy as np
import cv2
import keyboard
import os
import os.path
import time
import torch

from PIL import ImageGrab, Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QMainWindow

# DISK 경로
DISK = "D:"
# 녹스 실행경로
ADDRESS_BlueStack = "C:\Program Files\BlueStacks\Bluestacks.exe"
# 이미지, 모델, 디바이스 경로
WEBCAM_PATH = '0'
MODEL_PATH = DISK+"\\Timberman\\YOLOv3\\runs\\train\\8696\\weights\\best.pt"
# CUDA 경로
DEVICE_PATH = '0'

########################################################################################################################
# PyQt UI 클래스
########################################################################################################################

class Ui_MainWindow(QMainWindow, QObject):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(338, 305)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./img/TAEHAK.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setWhatsThis("")
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setWhatsThis("")
        self.centralwidget.setObjectName("centralwidget")
        self.OnOff = QtWidgets.QPushButton(self.centralwidget)
        self.OnOff.setGeometry(QtCore.QRect(256, 10, 75, 111))
        self.OnOff.setObjectName("OnOff")
        self.Train = QtWidgets.QPushButton(self.centralwidget)
        self.Train.setGeometry(QtCore.QRect(10, 160, 161, 71))
        self.Train.setObjectName("Train")
        self.Test = QtWidgets.QPushButton(self.centralwidget)
        self.Test.setGeometry(QtCore.QRect(170, 160, 161, 71))
        self.Test.setObjectName("Test")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(15, 130, 316, 25))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Model = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.Model.setObjectName("Model")
        self.horizontalLayout.addWidget(self.Model)
        self.ModelName = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.ModelName.setObjectName("ModelName")
        self.horizontalLayout.addWidget(self.ModelName)
        self.BrowseBtn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.BrowseBtn.setObjectName("BrowseBtn")
        self.horizontalLayout.addWidget(self.BrowseBtn)
        self.OpenClose = QtWidgets.QDialogButtonBox(self.centralwidget)
        self.OpenClose.setGeometry(QtCore.QRect(170, 240, 161, 23))
        self.OpenClose.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.OpenClose.setObjectName("OpenClose")
        self.Image = QtWidgets.QLabel(self.centralwidget)
        self.Image.setPixmap(QtGui.QPixmap("./img/TIMBERMAN.jpg"))
        self.Image.setGeometry(QtCore.QRect(15, 11, 231, 111))
        self.Image.setObjectName("Image")
        self.OnOff.raise_()
        self.Test.raise_()
        self.Train.raise_()
        self.horizontalLayoutWidget.raise_()
        self.OpenClose.raise_()
        self.Image.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 338, 21))
        self.menubar.setObjectName("menubar")
        self.menuInfo = QtWidgets.QMenu(self.menubar)
        self.menuInfo.setObjectName("menuInfo")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionCopyright = QtWidgets.QAction(MainWindow)
        self.actionCopyright.setObjectName("actionCopyright")
        self.menuInfo.addAction(self.actionCopyright)
        self.menubar.addAction(self.menuInfo.menuAction())

        self.retranslateUi(MainWindow)
        self.BrowseBtn.clicked.connect(self.browseSlot)
        self.OnOff.clicked.connect(self.powerSlow)
        self.Train.clicked.connect(self.trainSlot)
        self.Test.clicked.connect(self.testSlot)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Playing AI"))
        self.OnOff.setText(_translate("MainWindow", "Player\nOn"))
        self.Train.setText(_translate("MainWindow", "Train"))
        self.Test.setText(_translate("MainWindow", "Test"))
        self.Model.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Model</p></body></html>"))
        self.BrowseBtn.setText(_translate("MainWindow", "Browse"))
        self.menuInfo.setTitle(_translate("MainWindow", "Info"))
        self.actionCopyright.setText(_translate("MainWindow", "Copyright"))

########################################################################################################################
# UI 이벤트 함수
########################################################################################################################

    @pyqtSlot()
    def browseSlot(self):
        '''Qt5 모델파일 불러오기 이벤트'''
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')
        self.ModelName.setText(fname[0])
        self.ModelName.setReadOnly(True)
        return

    @pyqtSlot()
    def powerSlow(self):
        '''Nox 실행'''
        init()
        return

    @pyqtSlot()
    def trainSlot(self):
        '''Qt5 훈련버튼 이벤트(모니터링 시스템 실행)'''
        training()
        return

    @pyqtSlot()
    def testSlot(self):
        '''Qt5 테스트버튼 이벤트(모니터링 시스템 실행)'''
        #testing()

########################################################################################################################
# Main 관련 함수
########################################################################################################################

def init():
    '''녹스 실행 및 반환'''
    BlueStack = subprocess.Popen([ADDRESS_BlueStack, "BlueStack"], stdout = subprocess.PIPE)
    return BlueStack

def training():
    '''TRAIN 모듈'''
    detecting_capture()
    return

# def testing():
#     '''TEST 모듈'''
#     detecting_capture()
#     return

def detecting_capture():
    '''녹스 플레이화면 탐지'''
    os.chdir(DISK+'\\Timberman\\YOLOv3')
    subprocess.run('python detect.py --source '+WEBCAM_PATH+' --weights '+MODEL_PATH+' --device '+DEVICE_PATH, shell=True)
    return

def ternary(center_array):
    '''탐지화면 삼진화 -> 상태식(Domain) 생성 // x-axis :: 60 ++ 50, y-axis :: 0 ++ 320'''
    offset_str = ''
    for data in center_array:
        col_offset = data[1][0]//320+1 # [y] +1 -> 상태 혼동 방지 bias
        row_offset = (data[1][1] - 60) // 50 + 1 # [x] -60 -> 모니터링 화면과 YOLO모델 픽셀 차이 상쇄 // +1 -> 상태혼동방지
        offset_str += str(row_offset)+str(col_offset) # State => 동적 다변수 -> 변수 개수 정적화
    return offset_str