# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.Qt import QDesktopServices, QUrl
import os, sys
from datetime import datetime
import cv2
from json_loader import *
import json_loader
import re
from Editor import Editor
import time
import glob

'''
COCO Format
1	nose
2	left_eye
3	right_eye
4	left_ear
5	right_ear
6	left_shoulder
7	right_shoulder
8	left_elbow
9	right_elbow
10	left_wrist
11	right_wrist
12	left_hip
13	right_hip
14	left_knee
15	right_knee
16	left_ankle
17	right_ankle
'''

'''
To do list
1. 표시 이미지사이즈 확대축소랑 키포인트도 동일비율로 확대축소할것
2. hrnet 로딩 되는지 확인할것
3. 드롭다운리스트에 키포인트파일 생성하는 팝업창 따로만들기
4. 로딩완료되면 annotation file에 자동으로 등록되도록? (이건나중에)
5. 추가한 키포인트들 연결하기
6. 스타트버튼 안눌러도 어노테이션파일 로딩되면 스타트되도록 수정
7. 사람 bbox가 검출되지 않으면 리스트에서 볼때 자동으로 넘기도록하는 기능
8. 바운딩박스 그리는거 음수안나오게 고치기
9. 바운딩박스 그림밖에 찍는거 안되게 고치기
10. 나중에 로거코드 다 지우기
11. 포인트 이미지말고 타원그리기로 바꿀것
13. 포인트 사이에 선그리기

추가오류
painter update 문제 :업데이트될때 점이랑 박스 보여지게 고치기
'''

nowTime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

form_class = uic.loadUiType("lowResUI.ui")[0]

def search(dirname):
    filenames = os.listdir(dirname)
    myImageList = list()
    for filename in filenames:
        try:
            if filename.split('.')[-1] in ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']:

                full_filename = os.path.join(dirname, filename)
                myImageList.append(full_filename)

        except:
            continue

    return myImageList

class MyWindow(QMainWindow, QWidget,form_class):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

        self.keypointList = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow",
                             "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle","right_ankle", "bbox"]

        self.max_keypoint = len(self.keypointList) -1

        self.guide_img.setPixmap(QtGui.QPixmap("UI_image/UI_image.png"))
        self.Button_loadDIR.clicked.connect(self.open_text)
        self.Button_annotation.clicked.connect(self.clickAnnotation)
        self.Button_next.clicked.connect(self.showNextImage)
        self.Button_previous.clicked.connect(self.showPreviousImage)
        self.Button_start.clicked.connect(self.clickStartButton)
        self.imageDIR = ''
        self.imageLIST = list()
        self.imageINDEX = -1
        self.nowClickedPoint = 0 # init
        self.inputKeypointList= [[0]*3 for i in range(self.max_keypoint)]
        self.drawing = False
        self.backSeatPerson = False
        self.abnormal = False
        self.nowInvisibleKeypoint = False
        self.ix, self.iy = 0, 0
        self.iw, self.ih = 0,0
        self.bx, self.by = 0,0

        self.area = 0
        self.out_file_name = None
        self.now_image_list = []

        self.keypointButtonList = [self.NOSE_B, self.L_EYE_B,  self.R_EYE_B, self.L_EAR_B, self.R_EAR_B,
                                   self.L_SHLDER_B,self.R_SHLDER_B, self.L_ELBOW_B, self.R_ELBOW_B,
                                   self.L_WRIST_B, self.R_WRIST_B, self.L_HIP_B, self.R_HIP_B,
                                   self.L_KNEE_B, self.R_KNEE_B, self.L_ANKLE_B, self.R_ANKLE_B,
                                   self.Button_bbox]
        self.pointSpotList = [self.point_0, self.point_1, self.point_2,self.point_3,self.point_4,self.point_5,
                              self.point_6,self.point_7,self.point_8, self.point_9, self.point_10, self.point_11
                              , self.point_12, self.point_13, self.point_14, self.point_15, self.point_16]
        self.showInvisibleWhitePoint = True

        self.NOSE_B.clicked.connect(self.clickNose)
        self.L_EYE_B.clicked.connect(self.clickLeftEye)
        self.R_EYE_B.clicked.connect(self.clickRightEye)
        self.L_EAR_B.clicked.connect(self.clickLeftEar)
        self.R_EAR_B.clicked.connect(self.clickRightEar)
        self.L_SHLDER_B.clicked.connect(self.clickLeftShoulder)
        self.R_SHLDER_B.clicked.connect(self.clickRightShoulder)
        self.L_ELBOW_B.clicked.connect(self.clickLeftElbow)
        self.R_ELBOW_B.clicked.connect(self.clickRightElbow)
        self.Button_save.clicked.connect(self.clickSaveButton)
        self.Button_save_file.clicked.connect(self.clickSaveFileButton)
        self.Button_bbox.clicked.connect(self.clickBbox)
        self.person_id.valueChanged.connect(self.person_id_Changed)
        self.Button_remove.clicked.connect(self.removeButtonChange)
        self.Button_invisible.stateChanged.connect(self.invisible_Changed)
        self.Button_remove_one.clicked.connect(self.removeButtonOneChange)

        self.outFile = Editor()

        #### json 파일 경로 변수 ####
        self.json_path = ""

        #### json으로부터 가져온 dict 자료형 변수 ####
        self.json_dict = None

        self.point_0.setPixmap(QtGui.QPixmap("UI_image/point1.png"))
        self.point_1.setPixmap(QtGui.QPixmap("UI_image/point2.png"))
        self.point_2.setPixmap(QtGui.QPixmap("UI_image/point2.png"))
        self.point_3.setPixmap(QtGui.QPixmap("UI_image/point3.png"))
        self.point_4.setPixmap(QtGui.QPixmap("UI_image/point3.png"))
        self.point_5.setPixmap(QtGui.QPixmap("UI_image/point4.png"))
        self.point_6.setPixmap(QtGui.QPixmap("UI_image/point5.png"))
        self.point_7.setPixmap(QtGui.QPixmap("UI_image/point5.png"))
        self.point_8.setPixmap(QtGui.QPixmap("UI_image/point6.png"))
        self.point_9.setPixmap(QtGui.QPixmap("UI_image/point6.png"))
        self.point_10.setPixmap(QtGui.QPixmap("UI_image/point3.png"))
        self.point_11.setPixmap(QtGui.QPixmap("UI_image/point4.png"))
        self.point_12.setPixmap(QtGui.QPixmap("UI_image/point5.png"))
        self.point_13.setPixmap(QtGui.QPixmap("UI_image/point5.png"))
        self.point_14.setPixmap(QtGui.QPixmap("UI_image/point6.png"))
        self.point_15.setPixmap(QtGui.QPixmap("UI_image/point6.png"))
        self.point_16.setPixmap(QtGui.QPixmap("UI_image/point6.png"))
        for i in range(self.max_keypoint):
            self.pointSpotList[i].setVisible(False)

    def loadJsonData(self, now_file_name):
        retVal = False
        for content in self.json_dict.myjson["images"]:
            if str(content["file_name"]) == str(now_file_name) or str(''.join(filter(str.isdigit, now_file_name))) == (os.path.splitext(str(content["file_name"]) )[0]):
                #print(str(''.join(filter(str.isdigit, now_file_name))),(os.path.splitext(str(content["file_name"]) )[0]))
                for annotationContent in self.json_dict.myjson["annotations"]:
                    if annotationContent["image_id"] == content["id"] \
                            and self.now_image_list[int(self.person_id.value())-1] == annotationContent["id"]:
                        #and int(self.person_id.value()) == annotationContent["person_id"] :

                        self.area = annotationContent["area"]

                        for i in range(self.max_keypoint):
                            self.inputKeypointList[i][0] = (annotationContent["keypoints"])[i*3]
                            self.inputKeypointList[i][1] = (annotationContent["keypoints"])[i*3 +1]
                            self.inputKeypointList[i][2] = (annotationContent["keypoints"])[i*3 +2]

                            self.pointSpotList[i].setGeometry(QtCore.QRect(self.inputKeypointList[i][0] + self.guide_img_2.x()-5,  self.inputKeypointList[i][1] + self.guide_img_2.y()-5, 10, 10))


                        self.bx, self.by, self.iw, self.ih = annotationContent["bbox"]
                        print("bbox: ",self.bx, self.by, self.iw, self.ih)

                self.drawing = True

                retVal = True

        return retVal

    def refreshNowImageList(self):
        del self.now_image_list[:]

        for annotationContent in self.json_dict.myjson["annotations"]:
            now_id = int(''.join(filter(str.isdigit, os.path.basename(self.imageLIST[self.imageINDEX]))))
            if annotationContent["image_id"] == now_id:
                self.now_image_list.append(annotationContent["id"])

    def showNextImage(self):
        if self.imageINDEX == len(self.imageLIST) - 1:
            pass
        else:
            self.imageINDEX += 1
        self.guide_img_2.setPixmap(QtGui.QPixmap(self.imageLIST[self.imageINDEX]))
        #self.guide_img_2.setVisible(True)
        self.getImageData()

        self.ix, self.iy, self.iw, self.ih, self.bx, self.by = 0, 0, 0, 0, 0, 0
        for i in range(self.max_keypoint):
            self.pointSpotList[i].setVisible(False)
            for j in range(3):
                self.inputKeypointList[i][j] = 0  # init

        self.refreshNowImageList()
        if self.loadJsonData(os.path.basename(self.imageLIST[self.imageINDEX])) == True:
            for i in range(self.max_keypoint):
                if (self.inputKeypointList[i][0]) != 0 and (self.inputKeypointList[i][1] != 0) and (self.inputKeypointList[i][2] != 0)  :
                    self.pointSpotList[i].setVisible(True)
                elif (self.inputKeypointList[i][0]) != 0 and (self.inputKeypointList[i][1] != 0) and (self.inputKeypointList[i][2] == 0)  : #invisible point
                    if self.showInvisibleWhitePoint == True:
                        self.pointSpotList[i].setPixmap(QtGui.QPixmap("UI_image/point7.png"))
                    self.pointSpotList[i].setVisible(True)
                else :
                    self.pointSpotList[i].setVisible(False)

        else :
            #self.painter.eraseRect(30, 120, 640, 480)
            self.ix, self.iy, self.iw, self.ih, self.bx, self.by = 0, 0, 0, 0, 0, 0
            for i in range(self.max_keypoint):
                self.pointSpotList[i].setVisible(False)
                for j in range(3):
                    self.inputKeypointList[i][j] = 0 # init

        self.update()

    def showPreviousImage(self):
        if self.imageINDEX >= 1:
            self.imageINDEX -= 1
        else:
            self.imageINDEX = 0
        self.guide_img_2.setPixmap(QtGui.QPixmap(self.imageLIST[self.imageINDEX]))
        #self.guide_img_2.setVisible(True)
        self.getImageData()

        self.ix, self.iy, self.iw, self.ih, self.bx, self.by = 0, 0, 0, 0, 0, 0
        for i in range(self.max_keypoint):
            self.pointSpotList[i].setVisible(False)
            for j in range(3):
                self.inputKeypointList[i][j] = 0  # init

        self.refreshNowImageList()

        if self.loadJsonData(os.path.basename(self.imageLIST[self.imageINDEX])) == True:
            for i in range(self.max_keypoint):
                if (self.inputKeypointList[i][0]) != 0 and (self.inputKeypointList[i][1] != 0)  and (self.inputKeypointList[i][2] != 0) :
                    self.pointSpotList[i].setVisible(True)
                elif (self.inputKeypointList[i][0]) != 0 and (self.inputKeypointList[i][1] != 0) and (self.inputKeypointList[i][2] == 0)  : #invisible point
                    if self.showInvisibleWhitePoint == True:
                        self.pointSpotList[i].setPixmap(QtGui.QPixmap("UI_image/point7.png"))
                    self.pointSpotList[i].setVisible(True)
                else :
                    self.pointSpotList[i].setVisible(False)

        else :
            self.ix, self.iy, self.iw, self.ih, self.bx, self.by = 0, 0, 0, 0, 0, 0
            for i in range(self.max_keypoint):
                self.pointSpotList[i].setVisible(False)
                for j in range(3):
                    self.inputKeypointList[i][j] = 0 # init

        self.update()

    def open_text(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open File', os.getenv('HOME'), "Image files (*.jpg *.png *.bmp *.jpeg)" )

        if file_name[0] :
            self.imageDIR = file_name[0].split('/')[0:-1]
            #print('/'.join(self.imageDIR))

            list_image_names = os.listdir('/'.join(self.imageDIR))

            self.imageLIST = []

            for list_image_name in list_image_names:
                self.imageLIST.append('/'.join(self.imageDIR) + '/'+ list_image_name)

            #self.imageLIST = search('/'.join(self.imageDIR))
            #self.imageDIR.append('*.*')
            #self.imageLIST = glob.glob('/'.join(self.imageDIR))
            self.imageLIST.sort()
            #for image_name in self.imageLIST:
            #    image_name = image_name.replace('\\', '/')
            self.Line_load.setText('/'.join(self.imageDIR))


    def getImageData(self):
        self.image_name.setText("Now Image : <font color=\"#104E8B\">"+ os.path.basename(self.imageLIST[self.imageINDEX])+"</font>")

        self.image_src = cv2.imread(self.imageLIST[self.imageINDEX], cv2.IMREAD_COLOR)
        self.height, self.width, self.channel = self.image_src.shape
        self.image_size.setText("Width : <font color=\"#104E8B\">"+ str(self.width) +"</font>,  Height : <font color=\"#104E8B\">"+ str(self.height) +"</font>")

        self.point_0.setPixmap(QtGui.QPixmap("UI_image/point1.png"))
        self.point_1.setPixmap(QtGui.QPixmap("UI_image/point2.png"))
        self.point_2.setPixmap(QtGui.QPixmap("UI_image/point2.png"))
        self.point_3.setPixmap(QtGui.QPixmap("UI_image/point3.png"))
        self.point_4.setPixmap(QtGui.QPixmap("UI_image/point3.png"))
        self.point_5.setPixmap(QtGui.QPixmap("UI_image/point4.png"))
        self.point_6.setPixmap(QtGui.QPixmap("UI_image/point5.png"))
        self.point_7.setPixmap(QtGui.QPixmap("UI_image/point5.png"))
        self.point_8.setPixmap(QtGui.QPixmap("UI_image/point6.png"))
        self.point_9.setPixmap(QtGui.QPixmap("UI_image/point6.png"))
        self.point_10.setPixmap(QtGui.QPixmap("UI_image/point3.png"))
        self.point_11.setPixmap(QtGui.QPixmap("UI_image/point4.png"))
        self.point_12.setPixmap(QtGui.QPixmap("UI_image/point5.png"))
        self.point_13.setPixmap(QtGui.QPixmap("UI_image/point5.png"))
        self.point_14.setPixmap(QtGui.QPixmap("UI_image/point6.png"))
        self.point_15.setPixmap(QtGui.QPixmap("UI_image/point6.png"))
        self.point_16.setPixmap(QtGui.QPixmap("UI_image/point6.png"))

        for i in range(self.max_keypoint):
            self.pointSpotList[i].setVisible(False)

    def paintEvent(self,event):
        self.painter = QPainter()
        self.painter.begin(self)

        #if self.imageINDEX == -1:
        #    pass
        #else:
        #    self.painter.drawPixmap(self.guide_img_2.x(),self.guide_img_2.y(),QtGui.QPixmap(self.imageLIST[self.imageINDEX]))

        #self.painter.setPen(QPen(Qt.red, 4))
        #self.painter.drawRect(self.ix , self.iy , self.iw,self.ih )
        #self.painter.drawRect(self.bx + self.guide_img_2.x(), self.by + self.guide_img_2.y(), self.iw,self.ih )

        self.painter.setPen(QPen(Qt.red, 2))
        self.painter.drawLine(30, 230, 200, 50)
        #self.painter.drawEllipse(self.inputKeypointList[0][0] + self.guide_img_2.x()-3,  self.inputKeypointList[0][1] + self.guide_img_2.y()-3, 6, 6)
        #print("paintEvent")

        self.painter.end()
        #self.drawing = False

    def clickBbox(self):
        self.nowClickedPoint = self.max_keypoint
        self.nowButtonChange()

    def mouseReleaseEvent(self, event):
        if(event.x() < self.guide_img_2.x() or event.y() < self.guide_img_2.y()
                or event.x() > self.guide_img_2.x() + self.guide_img_2.width() or event.y() > self.guide_img_2.y() + self.guide_img_2.height())\
                or (self.ix < self.guide_img_2.x() or self.iy < self.guide_img_2.y()
                or self.ix > self.guide_img_2.x() + self.guide_img_2.width() or self.iy > self.guide_img_2.y() + self.guide_img_2.height()):
            self.statusbar.showMessage("범위를 벗어났습니다!")

        elif self.nowClickedPoint == self.max_keypoint:
            if event.button() == QtCore.Qt.LeftButton:
                print("Draw bbox")
                # cv2.rectangle(img, (ix, iy), (event.x(), event.y()), (0, 255, 0), -1)
                self.iw = event.x() - self.ix
                self.ih = event.y() - self.iy
                #self.paintEvent
                self.update()

    def mousePressEvent(self, event):
        txt = "클릭 위치 ; x={0},y={1}, in image={2},{3}".format(event.x(), event.y(),
                                                                event.x() - self.guide_img_2.x(), event.y() - self.guide_img_2.y())
        self.statusbar.showMessage(txt)

        if(event.x() < self.guide_img_2.x() or event.y() < self.guide_img_2.y()
                or event.x() > self.guide_img_2.x() + self.guide_img_2.width() or event.y() > self.guide_img_2.y() + self.guide_img_2.height()):
            print("Out of range!!")
            self.statusbar.showMessage("범위를 벗어났습니다!")
            self.ix, self.iy = event.x(), event.y()

        elif self.nowClickedPoint == self.max_keypoint:
            if event.button() == QtCore.Qt.LeftButton:
                self.iw, self.ih = 0, 0
                self.drawing = True
                self.ix, self.iy = event.x(), event.y()
                self.bx, self.by = event.x() - self.guide_img_2.x() , event.y() - self.guide_img_2.y()

        else:
            print("now button : ", self.keypointList[self.nowClickedPoint], "invisible?", self.nowInvisibleKeypoint)
            self.inputKeypointList[self.nowClickedPoint][0] = int(event.x() - self.guide_img_2.x())
            self.inputKeypointList[self.nowClickedPoint][1] = int(event.y() - self.guide_img_2.y())
            if self.nowInvisibleKeypoint == False:
                self.inputKeypointList[self.nowClickedPoint][2] = 1 # visible
            if self.nowInvisibleKeypoint == True:
                self.inputKeypointList[self.nowClickedPoint][2] = 0 # invisible

            #show point
            if self.nowClickedPoint == 0:
                self.point_0.setGeometry(QtCore.QRect(event.x()-5, event.y()-5, 10, 10))
                self.point_0.setPixmap(QtGui.QPixmap("UI_image/point1.png"))
                self.point_0.setVisible(True)
            if self.nowClickedPoint == 1:
                self.point_1.setGeometry(QtCore.QRect(event.x()-5, event.y()-5, 10, 10))
                self.point_1.setPixmap(QtGui.QPixmap("UI_image/point2.png"))
                self.point_1.setVisible(True)
            if self.nowClickedPoint == 2:
                self.point_2.setGeometry(QtCore.QRect(event.x()-5, event.y()-5, 10, 10))
                self.point_2.setPixmap(QtGui.QPixmap("UI_image/point2.png"))
                self.point_2.setVisible(True)
            if self.nowClickedPoint == 3:
                self.point_3.setGeometry(QtCore.QRect(event.x()-5, event.y()-5, 10, 10))
                self.point_3.setPixmap(QtGui.QPixmap("UI_image/point3.png"))
                self.point_3.setVisible(True)
            if self.nowClickedPoint == 4:
                self.point_4.setGeometry(QtCore.QRect(event.x()-5, event.y()-5, 10, 10))
                self.point_4.setPixmap(QtGui.QPixmap("UI_image/point3.png"))
                self.point_4.setVisible(True)
            if self.nowClickedPoint == 5:
                self.point_5.setGeometry(QtCore.QRect(event.x()-5, event.y()-5, 10, 10))
                self.point_5.setPixmap(QtGui.QPixmap("UI_image/point4.png"))
                self.point_5.setVisible(True)
            if self.nowClickedPoint == 6:
                self.point_6.setGeometry(QtCore.QRect(event.x()-5, event.y()-5, 10, 10))
                self.point_6.setPixmap(QtGui.QPixmap("UI_image/point5.png"))
                self.point_6.setVisible(True)
            if self.nowClickedPoint == 7:
                self.point_7.setGeometry(QtCore.QRect(event.x()-5, event.y()-5, 10, 10))
                self.point_7.setPixmap(QtGui.QPixmap("UI_image/point5.png"))
                self.point_7.setVisible(True)
            if self.nowClickedPoint == 8:
                self.point_8.setGeometry(QtCore.QRect(event.x()-5, event.y()-5, 10, 10))
                self.point_8.setPixmap(QtGui.QPixmap("UI_image/point6.png"))
                self.point_8.setVisible(True)
            if self.nowClickedPoint == 9:
                self.point_9.setGeometry(QtCore.QRect(event.x()-5, event.y()-5, 10, 10))
                self.point_9.setPixmap(QtGui.QPixmap("UI_image/point6.png"))
                self.point_9.setVisible(True)

            if self.nowInvisibleKeypoint == True:
                if self.showInvisibleWhitePoint == True:
                    self.pointSpotList[self.nowClickedPoint].setPixmap(QtGui.QPixmap("UI_image/point7.png"))


    def clickNose(self):
        self.nowClickedPoint = 0
        self.nowButtonChange()
    def clickLeftEye(self):
        self.nowClickedPoint = 1
        self.nowButtonChange()
    def clickRightEye(self):
        self.nowClickedPoint = 2
        self.nowButtonChange()
    def clickLeftEar(self):
        self.nowClickedPoint = 3
        self.nowButtonChange()
    def clickRightEar(self):
        self.nowClickedPoint = 4
        self.nowButtonChange()
    def clickLeftShoulder(self):
        self.nowClickedPoint = 6
        self.nowButtonChange()
    def clickRightShoulder(self):
        self.nowClickedPoint = 7
        self.nowButtonChange()
    def clickLeftElbow(self):
        self.nowClickedPoint = 8
        self.nowButtonChange()
    def clickRightElbow(self):
        self.nowClickedPoint = 9
        self.nowButtonChange()

    def keyPressEvent(self, e):

        if e.key() in [QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter]:#save
            print('Enter')
            self.clickSaveButton()
        if e.key() == QtCore.Qt.Key_Space: #Save
            print('Space')
            self.clickSaveButton()
        if e.key() == QtCore.Qt.Key_Q:
            self.nowClickedPoint = 4
            self.nowButtonChange()
        if e.key() == QtCore.Qt.Key_W:
            self.nowClickedPoint = 2
            self.nowButtonChange()
        if e.key() == QtCore.Qt.Key_E:
            self.nowClickedPoint = 0
            self.nowButtonChange()
        if e.key() == QtCore.Qt.Key_R:
            self.nowClickedPoint = 1
            self.nowButtonChange()
        if e.key() == QtCore.Qt.Key_T:
            self.nowClickedPoint = 3
            self.nowButtonChange()
        if e.key() == QtCore.Qt.Key_A:
            self.nowClickedPoint = 9
            self.nowButtonChange()
        if e.key() == QtCore.Qt.Key_S:
            self.nowClickedPoint = 7
            self.nowButtonChange()
        if e.key() == QtCore.Qt.Key_D:
            self.nowClickedPoint = 5
            self.nowButtonChange()
        if e.key() == QtCore.Qt.Key_F:
            self.nowClickedPoint = 6
            self.nowButtonChange()
        if e.key() == QtCore.Qt.Key_G:
            self.nowClickedPoint = 8
            self.nowButtonChange()
        if e.key() == QtCore.Qt.Key_B:
            self.nowClickedPoint = 10
            self.nowButtonChange()
            self.clickBbox()
        if e.key() == QtCore.Qt.Key_K:
            if self.person_id.value() > 1 :
                new_person_id = self.person_id.value() -1
                self.person_id.setValue(new_person_id)
        if e.key() == QtCore.Qt.Key_I:
            if self.person_id.value() < 5 :
                new_person_id = self.person_id.value() + 1
                self.person_id.setValue(new_person_id)
        if e.key() == QtCore.Qt.Key_L:
            self.showNextImage()
        if e.key() == QtCore.Qt.Key_J:
            self.showPreviousImage()
        if e.key() == QtCore.Qt.Key_V:
            self.Button_invisible.setCheckState(not (self.Button_invisible.isChecked()))
            self.invisible_Changed()
        if e.key() == QtCore.Qt.Key_C:
            self.removeButtonOneChange()


    def nowButtonChange(self):
        self.keypointButtonList[self.nowClickedPoint].setChecked(True)
        for i in range(11):
            if i != self.nowClickedPoint:
                self.keypointButtonList[i].setChecked(False)

        self.statusbar.showMessage("현재 선택된 항목 : " + self.keypointList[self.nowClickedPoint])

    def removeButtonChange(self):
        self.ix, self.iy, self.iw, self.ih, self.bx, self.by = 0, 0, 0, 0, 0, 0
        for i in range(self.max_keypoint):
            self.pointSpotList[i].setVisible(False)
            for j in range(3):
                self.inputKeypointList[i][j] = 0  # init

    def removeButtonOneChange(self):
        if self.nowClickedPoint == self.max_keypoint: #bounding box
            self.ix, self.iy, self.iw, self.ih, self.bx, self.by = 0, 0, 0, 0, 0, 0

        elif self.nowClickedPoint < self.max_keypoint:
            self.pointSpotList[self.nowClickedPoint].setVisible(False)
            for j in range(3):
                self.inputKeypointList[self.nowClickedPoint][j] = 0  # init

    def person_id_Changed(self):
        #self.guide_img_2.setPixmap(QtGui.QPixmap(self.imageLIST[self.imageINDEX]))
        self.getImageData()
        #self.person_id.value()

        self.ix, self.iy, self.iw, self.ih, self.bx, self.by = 0, 0, 0, 0, 0, 0
        for i in range(self.max_keypoint):
            self.pointSpotList[i].setVisible(False)
            for j in range(3):
                self.inputKeypointList[i][j] = 0  # init

        if self.loadJsonData(os.path.basename(self.imageLIST[self.imageINDEX])) == True :
            for i in range(self.max_keypoint):
                if (self.inputKeypointList[i][0]) != 0 and (self.inputKeypointList[i][1] != 0)  and (self.inputKeypointList[i][2] != 0):
                    self.pointSpotList[i].setVisible(True)

                elif (self.inputKeypointList[i][0]) != 0 and (self.inputKeypointList[i][1] != 0) and (self.inputKeypointList[i][2] == 0):  # invisible point
                    if self.showInvisibleWhitePoint == True:
                        self.pointSpotList[i].setPixmap(QtGui.QPixmap("UI_image/point7.png"))
                    self.pointSpotList[i].setVisible(True)
                else :
                    self.pointSpotList[i].setVisible(False)

        else :
            #self.painter.eraseRect(30, 120, 640, 480)
            self.ix, self.iy, self.iw, self.ih, self.bx, self.by = 0, 0, 0, 0, 0, 0
            for i in range(self.max_keypoint):
                self.pointSpotList[i].setVisible(False)
                for j in range(3):
                    self.inputKeypointList[i][j] = 0 # init


    def invisible_Changed(self):
        print("invisible_Changed ", self.Button_invisible.isChecked())
        self.nowInvisibleKeypoint = self.Button_invisible.isChecked()


    def clickAnnotation(self):
        self.out_file_name = QFileDialog.getOpenFileName(self, 'Open File', os.getenv('HOME'), "JSON files (*.json)" )

        if self.out_file_name[0]:
            self.Line_annotation.setText(self.out_file_name[0])

        self.clickStartButton()

    def clickStartButton(self):
        if self.imageDIR != '':
            self.statusbar.showMessage("Annotation을 시작합니다.")

            if self.out_file_name == None:
                self.json_dict = COCO2017_annotation_json()
            else:
                self.json_dict = COCO2017_annotation_json(self.out_file_name[0])

            self.showNextImage()


        elif self.imageDIR == '':
            self.statusbar.showMessage("먼저 폴더를 설정해주세요.")

    def clickSaveButton(self):
        if self.json_dict == None :
            self.statusbar.showMessage("'Start' 버튼을 먼저 클릭하고 진행해주세요.")
            return

        else :
            content_images_dict = {
                "license": 4,
                "file_name": os.path.basename(self.imageLIST[self.imageINDEX]),
                "height": self.height,
                "width": self.width,
                "date_captured": str(datetime.now()),
                "id":  int(''.join(filter(str.isdigit,(os.path.splitext(os.path.split(self.imageLIST[self.imageINDEX])[1])[0])))),#os.path.splitext(os.path.split(self.imageLIST[self.imageINDEX])[1])[0],
                "abnormal": self.abnormal
            }

            keypointList = sum(self.inputKeypointList, [])

            num_keypoints = 0
            for i in range(self.max_keypoint):
                if keypointList[i] > 0 :
                    num_keypoints += 1

            content_annotations_dict = {
                "num_keypoints": num_keypoints,
                "image_id": int(''.join(filter(str.isdigit,(os.path.splitext(os.path.split(self.imageLIST[self.imageINDEX])[1])[0])))),
                "bbox": [self.bx , self.by , self.iw,self.ih],
                "category_id" : 1,
                "keypoints": keypointList,
                "id": int(''.join(filter(str.isdigit,(os.path.splitext(os.path.split(self.imageLIST[self.imageINDEX])[1])[0] + str(self.person_id.value()))))),#os.path.splitext(os.path.split(self.imageLIST[self.imageINDEX])[1])[0],
			    "area": self.area,
                "iscrowd": 0
            }

            # image, annotation 정보를 json dictionary에 추가(내부적으로 비교)
            self.json_dict.set_IMAGES(content_images_dict)
            self.json_dict.set_ANNOTATIONS(content_annotations_dict)

            for i in range(self.max_keypoint):
                self.pointSpotList[i].setVisible(False)
                for j in range(3):
                    self.inputKeypointList[i][j] = 0 # init

    def clickSaveFileButton(self):
        self.statusbar.showMessage("저장중입니다. 잠시만 기다려주세요...")
        self.json_dict.saver(self.out_file_name[0])
        self.statusbar.showMessage("저장이 완료되었습니다. File : " + self.out_file_name[0])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    #app.setStyle('Fusion') #['Breeze', 'Oxygen', 'QtCurve', 'Windows', 'Fusion']
    #pal = QPalette()
    #w = QWidget()
    #pal.setColor(QPalette.Background, Qt.red)
    #pal.setColor(w.backgroundRole(), Qt.red)
    #w.setPalette(pal)
    #pal.setAutoFillBackground(True)
    #setPalette(pal)

    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
