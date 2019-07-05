'''
课程设计作业Task2
作者：陈昂
时间：2019.7.5  14:56
版权所有，盗版必究
https://github.com/MCLBHLSY/Chen
注：本程序实现功能是将faceImages目录下的所有文件转化成灰度图片，并依照原
    先的分类方式存到faceImageGray目录下，由于程序在创建路径之前，并没有判
    断此路径是否存在，所有此程序只能运行一次，第二次运行会报错，文件夹已存
    在，若想第二次运行，需要把faceImageGray目录下的所有文件删除即可，或者
    在os.makedirs(saved_path)这句之前用os.path.exists(saved_path)判断一下
    是否存在此文件夹，不存在时候再进行创建
    并且其中两个循环长度取得固定值10，600，如有特殊需要需在前面加上判断文
    件夹或文件数量的语句，可以实现动态适应。
'''
import os
import cv2
first_list = os.listdir('./faceImages/')
for i in range (0,10):
    saved_path = r'./faceImageGray/' + first_list[i]
    os.makedirs(saved_path)
    for j in range(0,600):
        image = cv2.imread('./faceImages/'+first_list[i]+'/'+str(j)+'.jpg')

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow(first_list[i],image_gray)
        saved_name = '/'+str(j)+'.jpg'
        cv2.imwrite(saved_path+saved_name, image_gray)
        cv2.waitKey(1)
