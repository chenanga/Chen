'''
课程设计作业  小实验:实时面部检测，检测摄像区域内人数，并用蓝色小框标记出面部
作者：陈昂
时间：2019.7.7  12:36
版权所有，盗版必究
https://github.com/MCLBHLSY/Chen
（mxnet_mtcnn模型引用自https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection.git，需要把该文件下载下来后放到和TASK.py同路径下，
然后把文件夹名称修改为mxnet_mtcnn_face_detection，并且把其文件夹下的mtcnn_detector.py第九行的# from itertools import izip替换成izip = zip）
'''

import cv2
import sys
import time
from mxnet_mtcnn_face_detection.mtcnn_detector import MtcnnDetector
import mxnet as mx
print(chr(84),chr(104),chr(105),chr(115),chr(32),chr(98),chr(101),chr(108),chr(111),chr(110),chr(103),chr(115),chr(32),chr(116),chr(111),chr(32),chr(67),chr(104),chr(101),chr(110),chr(32),chr(65),chr(110),chr(103),chr(46))
cap = cv2.VideoCapture(0)
if __name__ == '__main__':
    detector=MtcnnDetector(model_folder="./mxnet_mtcnn_face_detection/model", ctx=mx.cpu(0), num_worker=4, accurate_landmark=False)
    while True:
        ret, frame = cap.read()
        cv2.imshow('get_face', frame)
        try:
            res = detector.detect_face(frame)
            facebox = res[0]

            for b in facebox:
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
                sys.stdout.write("\r" + format('摄像区域检测到%d人'%len(facebox)))
        except:
            sys.stdout.write("\r" + format('摄像区域未检测到人脸'))
            sys.stdout.flush()  # 防止堵塞
            time.sleep(0.0001)
        cv2.imshow('get_face', frame)
        cv2.waitKey(1)
