# -*- coding: utf-8 -*-
"""
原 1_0CrnnDataPreprocess_3.py
对输入CRNN网络的数据进行预处理，输入是原始的偏心摄影验光的图片，输出一份是记录有眼瞳位置信息的一组图片，另一份是一组图片
1、第一帧图像使用surf特征点检测，粗略定位瞳孔位置
2、根据粗略定位的瞳孔位置确定一块区域，进行圆拟合

3、剔除模糊图像和瞳孔比较小的图像
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import self_def
import csv
import os
import pandas as pd
import time
plt.ion() #开启interactive mode

def main_preprocess():
    '''
    为什么会把它封装到函数里面呢，这是为了当找不到眼瞳的时候，可以直接跳过这一组图像
    :return:
    '''
    print(imgs_num)

    # 两个空列表，分别存放左眼和右眼的裁剪图片
    img_cut_left = []
    img_cut_right = []

    # 两个空列表，存放左右眼截取的斜率数据 #这些列表要记得清空的
    lines_left = []
    lines_right = []

    # 查看圆拟合的结果
    canny_edge = []

    #  把提取出的瞳孔位置，及近视度数存储到该.csv文件中
    file_handle = open('CenterPositionData.csv', 'a+', newline='')  # 不覆盖原数据的方式是a+ 覆盖原数据的方式是w
    writer = csv.writer(file_handle)
    position_write = []
    position_write.append(name_save)
    position_write.append(eye_paramater)
    position_write.append([str(imgs_num)])

    imgs_path = "F:/data/屈光/图像数据/" + str(people_name) + '/' + str(imgs_num)

    #查看是否有该文件夹
    try:
        img_name_list = os.listdir(imgs_path) #
    except FileNotFoundError:
        print("没有该学生文件夹  ")
        return

    if len(img_name_list) != 12:
        print("文件夹中图像数目不对")
        return

    img_name_list.sort(key=lambda x: int(x[:-4]))  # 将'.jpg'左边的字符转换成整数型进行排序

    img_num = 0
    for img_name in img_name_list:
        # print(i+1)
        img_num += 1  # 处理第几幅图片 #15
        img_path = os.path.join(imgs_path, img_name)
        try:
            # 下图方法可以解决中文路径的问题
            img = cv2.imdecode(
                np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)  #
        except FileNotFoundError:
            print("没有找到文件  ")
            return
        img_src = img.copy()

        if img_num == 1:
            try:
                centers = self_def.FindBrightSpot_3(img)
            except ValueError:
                print("没有找到瞳孔 ")
                return
            position_write.append([centers[0, 0], centers[0, 1], centers[1, 0], centers[1, 1]])
            position_write = [x for listt in position_write for x in listt]  # 这样可以遍历所有的列表元素了
        # print(i, centers[0])
        branch_num = 0

        # 提取出瞳孔的外部边缘

        # 内部边缘 #这是对全局进行的边缘提取，后续可以改成只对内部进行边缘提取
        src = cv2.GaussianBlur(img, (5, 5), 0)
        Canny_img_1 = cv2.Canny(src, 5, 15, (5, 5))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        Canny_img_1 = cv2.dilate(Canny_img_1, kernel)

        # 外部边缘
        equalize = cv2.equalizeHist(img)  # 直方图均衡化
        media_blur = cv2.medianBlur(equalize, 5)
        gaussian_blur = cv2.GaussianBlur(media_blur, (5, 5), 0)
        Canny_img_2_src = cv2.Canny(gaussian_blur, 90, 180, (11, 11))
        Canny_img_2 = cv2.dilate(Canny_img_2_src, kernel)

        # 综合内外边缘
        temp_result = cv2.multiply(Canny_img_1, Canny_img_2)
        result = cv2.multiply(temp_result, Canny_img_2_src)


        '''plt.figure()
        plt.imshow(Canny_img_1,'gray')
        plt.figure()
        plt.imshow(result,'gray')
        plt.figure()
        plt.imshow(Canny_img_2_src,'gray')
        plt.show()'''

        for center in centers:
            branch_num += 1
            edge_cut = result[center[1] - EYE_AREA:center[1] + EYE_AREA,
                       center[0] - EYE_AREA:center[0] + EYE_AREA]  # 初步截取图片，为了保险，比原来规定好的图片要大一些
            x, y, r = self_def.PupilEdgeExtract_2(edge_cut, last_center=[EYE_AREA, EYE_AREA, EYE_AREA])  # 拟合圆

            # 更改中心值，原来是以亮斑作为中心，现在改成以拟合的圆形瞳孔区域作为中心
            # if np.linalg.norm(np.array([x,y])-np.array([EYE_AREA,EYE_AREA])) < 15 :
            center[1] = center[1] + y - EYE_AREA
            center[0] = center[0] + x - EYE_AREA
            img_cut = (img_src[center[1] - EYE_AREA:center[1] + EYE_AREA,
                       center[0] - EYE_AREA:center[0] + EYE_AREA]).copy()  # 瞳孔拟合圆中心作为新的中心，去截取图片
            # cv2.imwrite('EyeArea_2/'+str(img_num) + '_' + str(branch_num) + '.bmp', img_cut)
            # cv2.circle(img_cut, (EYE_AREA, EYE_AREA), r, (255, 255, 255))

            # 截取后的图片按照左右眼分开存储到两个空列表中
            if branch_num == 1:  # 左眼
                img_cut_left.append(img_cut)
            else:
                img_cut_right.append(img_cut)

    print("拟合圆的直径为", 2 * r)
    # 显示截取之后的左右眼图片，确保显示是正确的
    plt.figure(figsize=(10, 10))  # 示左眼
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.imshow(img_cut_left[i], 'gray')
        # cv2.imwrite('EyeArea/' + people_name+'/'+str(imgs_num) +'/'+str(i)+ '_1'  + '.bmp', img_cut_left[i])

    '''plt.figure(2)  # 显示右眼
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(img_cut_right[i], 'gray')'''

    #计算图片相似度 用于剔除中间闭眼的图像
    '''img_sim = []
    for i in range(11):
        img1 = img_cut_left[i]
        #cv2.normalize(img1, img1) 别归一化了，直接一样了
        img1 = (img1[EYE_AREA - 32:EYE_AREA + 32, EYE_AREA - 32:EYE_AREA + 32,]).copy()
        img1 = np.array(img1,dtype=int)
        img2 = img_cut_left[i+1]
        img2 = (img2[EYE_AREA - 32:EYE_AREA + 32, EYE_AREA - 32:EYE_AREA + 32, ]).copy()
        img2 = np.array(img2, dtype=int)
        simi = img2 - img1
        simi = np.sum(simi**2)
        #使用哈希不行
        #simi = self_def.pHash(img1, img2)[0]
        #hash1 = self_def.pHash(img1)
        #hash2 = self_def.pHash(img2)
        #simi = self_def.cmpHash(hash1,hash2)
        #simi = self_def.calculate(img1, img2)[0]
        img_sim.append(simi)
    print(img_sim)'''

    # 图片延时一段时间后自动关闭
    plt.draw()
    plt.pause(0.5)
    plt.close()

    # plt.show()

    save_flag = input("是否保存图片")

    if save_flag == '':
        # 保存图片位置信息
        writer.writerow(position_write)  # 保存瞳孔位置信息

        # 保存图片
        path = 'F:/YizhouZhuruiEyeArea/' + name_save[0] + '/' + str(imgs_num) + '/1/'  # 左眼保存路径
        folder = os.path.exists(path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        for i in range(12):
            cv2.imwrite(path + str(i + 1) + '.bmp',
                        img_cut_left[i])

        path = 'F:/YizhouZhuruiEyeArea/' + name_save[0] + '/' + str(imgs_num) + '/2/'  # 右眼保存路径
        folder = os.path.exists(path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        for i in range(12):
            cv2.imwrite(path + str(i + 1) + '.bmp',
                        img_cut_right[i])


if __name__ == '__main__':
    data = pd.read_csv('F:/data/屈光/data_xinjiang.csv',engine='python',encoding='gbk').values

    #data = pd.read_csv('2_data_diopter.csv',engine='python',encoding='gbk').values
    # 提取学号数据
    m, n = np.shape(data)
    #for i in range(m):
    #    data[i,0] = data[i,0][1:]

    count = 5148
    #表示着第几个人名 #713

    for people_num in range(m-count):
        print("这是第 ",count+people_num, " 个人的图片")
        people_name = data[count+people_num,0]
        print(people_name)
        name_save = [data[count+people_num,0]]
        eye_paramater = data[count+people_num,1:7].tolist()
        imgs_numbers = ['LEFT']  # 选择那些清晰的图片组 #1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20

        EYE_AREA = 128
        EYE_AREA = int(EYE_AREA / 2)

        # 遍历20组图片， 查看截取的对不对
        for imgs_num in imgs_numbers:
            main_preprocess()

