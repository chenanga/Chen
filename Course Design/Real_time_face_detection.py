'''
课程设计作业  小实验:实时面部检测
作者：陈昂
时间：2019.7.7  12:09
版权所有，盗版必究
https://github.com/MCLBHLSY/Chen
（mxnet_mtcnn模型引用自https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection.git，需要把该文件下载下来后放到和TASK.py同路径下，
然后把文件夹名称修改为mxnet_mtcnn_face_detection，并且把其文件夹下的mtcnn_detector.py第九行的# from itertools import izip替换成izip = zip）
'''
import cv2
from mxnet_mtcnn_face_detection.mtcnn_detector import MtcnnDetector
import mxnet as mx
cap = cv2.VideoCapture(0)
if __name__ == '__main__':
    detector=MtcnnDetector(model_folder="./mxnet_mtcnn_face_detection/model", ctx=mx.cpu(0), num_worker=4, accurate_landmark=False)
    while True:
        ret, frame = cap.read()
        cv2.imshow('get_face', frame)
        res = detector.detect_face(frame)  #进行面部检测，得到面部点坐标
        facebox = res[0]
        for b in facebox:
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)  #在原始图像中用框把人脸标记出来，(0, 0, 255)这个表示框的颜色，后面的 1表示线的宽度
        cv2.imshow('get_face', frame)
        cv2.waitKey(1)
