#转载请注明来源https://github.com/MCLBHLSY/Chen
import cv2
import os
import time

print("正在初始化摄像头,请稍后...")
cap = cv2.VideoCapture(0)
while True:
    name = input("初始化成功！\n请输入姓名拼音（输入后按下回车采集开始）：")
    print("采集即将开始（每隔0.2s拍摄一张，共计600张）")
    savedpath = r'./faceImages/' + name
    os.makedirs(savedpath)
    print("人脸信息子文件夹创建成功")
    i = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('data_read', frame)
        savedname = '/'  + str(i) +  '.jpg'
        k = cv2.waitKey(1) & 0xFF
        cv2.imwrite(savedpath + savedname, frame)
        i += 1
        if i>=600:
            break
        print('%s.jpg successful'%(i))
        time.sleep(0.2)#延时0.2s,每隔0.2s拍摄一张
    print("继续请按c,退出请按q")
    flag = input()
    if flag=='q':
        break
cap.release()
cv2.destroyAllWindows()
