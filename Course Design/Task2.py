'''
课程设计作业Task2
作者：陈昂
时间：2019.7.7  0:21
版权所有，盗版必究
https://github.com/MCLBHLSY/Chen
注：   ①本程序实现功能是将faceImages目录下的所有文件先进行面部检测（mxnet_mtcnn）再转化成
    灰度图片，并依照原先的分类方式存到faceImageGray目录下，由于程序在创建路径之前，并没有判
    断此路径是否存在，所有此程序只能运行一次，第二次运行会报错，文件夹已存在，若想第二次运行，
    需要把faceImageGray目录下的所有文件删除即可，或者在os.makedirs(saved_path)这句之前用
    os.path.exists(saved_path)判断一下是否存在此文件夹，不存在时候再进行创建。
    （mxnet_mtcnn模型引用自https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection.git，
    需要把该文件下载下来后放到和TASK.py同路径下，然后把文件夹名称修改为mxnet_mtcnn_face_detection，并且把其文件夹下的mtcnn_detector.py第九行的# from itertools import izip替换成izip = zip）
       ②并且其中两个循环长度取得固定值10，600，如有特殊需要需在前面加上判断文件夹或文件数量
       的语句，可以实现动态适应。
       ③程序最后实现了进度条功能，因为程序执行时间较长，此处设置了进度条，方便用户观看。
    ！！！④由于数据集采集时候部分照片模糊或者光线黑暗导致算法不能检测到面部，从而部分程序会
    报错，所以在程序的40,41,42,43,44行添加了一个判断，判别是否有值，也即为当前数据照片是否检测到脸
    部，若没检测到，则用上一次的照片数据为这次所用，避免程序出错。若照片集采集时候没有出现模
    糊或者光线暗等情况，则不需要这三行语句。（当然，如果是第一张数据模糊无法识别到，则仍会报错）
    同时由于里面多了几个照片数据的复制语句，所以程序运行会变慢，如果照片数据集完整则可删除这些行
'''
import os
import time
import cv2
import sys
from mxnet_mtcnn_face_detection.mtcnn_detector import MtcnnDetector
import mxnet as mx
if __name__ == '__main__':    #因为下面有分部语句是多线程，多进程需要在main函数中运行
    detector=MtcnnDetector(model_folder="./mxnet_mtcnn_face_detection/model", ctx=mx.cpu(0), num_worker=4, accurate_landmark=False)
    print(chr(84),chr(104),chr(105),chr(115),chr(32),chr(98),chr(101),chr(108),chr(111),chr(110),chr(103),chr(115),chr(32),chr(116),chr(111),chr(32),chr(67),chr(104),chr(101),chr(110),chr(32),chr(65),chr(110),chr(103),chr(46))
    first_list = os.listdir('./faceImages/')   #faceImages目录下所有文件夹的名称
    a=0 #程序执行的循环次数，本程序总循环次数6000次
    for i in range (0,10):
        saved_path = r'./faceImageGray/' + first_list[i]
        os.makedirs(saved_path)
        for j in range(0,600):
            image = cv2.imread('./faceImages/'+first_list[i]+'/'+str(j)+'.jpg')
            res = detector.detect_face(image)
            if res is None:
                res = res1    #此处只是为了让程序不报错
                image = image1   #此处为了输出错误的上一张照片的输出
            image1 = image
            res1 = res  # 保存上次输出的图像，如果这次图像由于环境等因素未检测到人脸，就把上次的赋值给这次，防止程序出错
            facebox = res[0]   #此处默认数据集照片中只存在一张人脸，忽略其他的，如需要其他的，这里对其进行遍历即可
            face_image = image[int(facebox[0, 1]):int(facebox[0, 3]), int(facebox[0, 0]):int(facebox[0, 2])]
            image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            # image_gray1 = image_gray
            cv2.imshow(first_list[i],image_gray)
            saved_name = '/'+str(j)+'.jpg'
            # if res2 is None:
            #     image_gray = image_gray1     ##此处是当当前照片模糊时候，把上次的图片复制给这次
            cv2.imwrite(saved_path+saved_name, image_gray)
            cv2.waitKey(1)
            a=a+1
            percent= a/6000     #程序当前进行的进度

            sys.stdout.write("\r"+format('当前进度%.2f%%' % (percent * 100)))#显示进度
            sys.stdout.flush()   #防止堵塞
            time.sleep(0.0001)

        cv2.destroyAllWindows()
