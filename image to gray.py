import cv2
import os
import numpy as np

file_root = 'C:/Users/111/Desktop/data/jinzhou/oringe/channel11/'  # 当前文件夹下的所有图片
file_list = os.listdir(file_root)
n = 1


sort_num_list = []
for file in file_list:
    sort_num_list.append(int(file.split('.jpg')[0])) #去掉前面的字符串和下划线以及后缀，只留下数字并转换为整数方便后面排序
    sort_num_list.sort() #然后再重新排序

sorted_file = []
for sort_num in sort_num_list:
    for file in file_list:
        if str(sort_num) == file.split('.jpg')[0]:
            sorted_file.append(file)


for img_name in sorted_file:
    img_path = file_root + img_name

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('1',image)
    #
    M = cv2.medianBlur(image, 13)  # 中值滤波


    #N = cv2.resize(M, [320,320], dst=None, fx=None, fy=None, interpolation=cv2.INTER_LANCZOS4)

    # 图像归一化
    fI = M / 255.0
    # 伽马变换
    gamma = 1.5
    O = np.power(fI, gamma)
    cv2.imshow('I', O)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    #cv2.imwrite(r'C:\Users\111\Desktop\data\gamma\origin\rgb\-5D\rgb-{}.jpg'.format(n), image)
    cv2.imwrite(r'C:\Users\111\Desktop\data\jinzhou\gamma\train\GRAY\channel11\{}.jpg'.format(n), O*255)
    n = n+1