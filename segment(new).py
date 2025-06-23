import cv2
import matplotlib.pyplot as plt
import numpy as np
import self_def
import csv
import os
import pandas as pd
import time
plt.ion() #开启interactive mode
from scipy import stats


def calculatepuiple(pupilcenters):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #img = img[240:1199, 0:1599]

    img_src = img.copy()

    # if imgs_num == 1:
    try:
        centers = self_def.FindBrightSpot_3(img)
    except ValueError:
        #print("没有找到瞳孔 ")
        return

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

    for center in centers:

        edge_cut = result[center[1] - EYE_AREA:center[1] + EYE_AREA,
                   center[0] - EYE_AREA:center[0] + EYE_AREA]  # 初步截取图片，为了保险，比原来规定好的图片要大一些

        try:
            x, y, r = self_def.PupilEdgeExtract_2(edge_cut, last_center=[EYE_AREA, EYE_AREA, EYE_AREA])  # 拟合圆
            # #
            # # # 更改中心值，原来是以亮斑作为中心，现在改成以拟合的圆形瞳孔区域作为中心
            # # #if np.linalg.norm(np.array([x,y])-np.array([EYE_AREA,EYE_AREA])) < 15 :
            center[1] = center[1] + y - EYE_AREA
            center[0] = center[0] + x - EYE_AREA
        except Exception as e:
            print(e)
            print(path2)
            return


    pupilcenters.append(centers)
    return

def main_preprocess(centers):
    '''
    为什么会把它封装到函数里面呢，这是为了当找不到眼瞳的时候，可以直接跳过这一组图像
    :return:
    '''
#    print(imgs_num)

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

    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    # img = img[240:1199,0:1599]
    # cv2.imshow('1',img)
    # cv2.waitKey(10)
    img_src = img.copy()

    try:
        position_write.append([centers[0, 0], centers[0, 1], centers[1, 0], centers[1, 1]])
        position_write = [x for listt in position_write for x in listt]  # 这样可以遍历所有的列表元素了
        # print(i, centers[0])
    except Exception as e:
        print(e)
        print(path2)

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

    for center in centers:
        branch_num += 1
        # edge_cut = result[center[1] - EYE_AREA:center[1] + EYE_AREA,
        #               center[0] - EYE_AREA:center[0] + EYE_AREA]  # 初步截取图片，为了保险，比原来规定好的图片要大一些

        # try:
        #     x, y, r = self_def.PupilEdgeExtract_2(edge_cut, last_center=[EYE_AREA, EYE_AREA, EYE_AREA])  # 拟合圆
        # # #
        # # # # 更改中心值，原来是以亮斑作为中心，现在改成以拟合的圆形瞳孔区域作为中心
        # # # #if np.linalg.norm(np.array([x,y])-np.array([EYE_AREA,EYE_AREA])) < 15 :
        #     center[1] = center[1] + y - EYE_AREA
        #     center[0] = center[0] + x - EYE_AREA
        # except Exception as e:
        #     print(e)
        #     print(path2)
        #     return


        img_cut = (img_src[center[1] - EYE_AREA:center[1] + EYE_AREA,
                       center[0] - EYE_AREA:center[0] + EYE_AREA]).copy()  # 瞳孔拟合圆中心作为新的中心，去截取图片

        # cv2.imwrite('EyeArea_2/'+str(img_num) + '_' + str(branch_num) + '.bmp', img_cut)
        # cv2.circle(img_cut, (EYE_AREA, EYE_AREA), r, (255, 255, 255))

        # 截取后的图片按照左右眼分开存储到两个空列表中
        if branch_num == 1:  # 左眼
            img_cut_left.append(img_cut)
        else:
            img_cut_right.append(img_cut)

    #print("拟合圆的直径为", 2 * r)
    # 显示截取之后的左右眼图片，确保显示是正确的
    #plt.figure(figsize=(10, 10))  # 示左眼
    #for i in range(12):
    #    plt.subplot(3, 4, i + 1)
    #plt.imshow(img_cut_left[0], 'gray')
        # cv2.imwrite('EyeArea/' + people_name+'/'+str(imgs_num) +'/'+str(i)+ '_1'  + '.bmp', img_cut_left[i])



        # 保存图片位置信息
    writer.writerow(position_write)  # 保存瞳孔位置信息

    # 保存图片
    pathl = 'F:/tongkongyingguangdian/' +  '1/' + name0 +'/' + name2 +'/' # 左眼保存路径
    folder = os.path.exists(pathl)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(pathl)  # makedirs 创建文件时如果路径不存在会创建这个路径
    #for i in range(12):
    cv2.imwrite(pathl + str(count) + '.bmp',
                        img_cut_left[0])

    # pathr = 'F:/moniyandingzhidata/' + '2/' + name0 + '/'+ name2 +'/'  # 右眼保存路径
    # folder = os.path.exists(pathr)
    # if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
    #     os.makedirs(pathr)  # makedirs 创建文件时如果路径不存在会创建这个路径
    #     #for i in range(12):
    # cv2.imwrite(pathr + str(count) + '.bmp',
    #                     img_cut_right[0])

    return

if __name__ == '__main__':

    path1 = ("F:\\tongkongyingguan1")  # 包含四类的文件夹路径
    for folder_name0 in os.listdir(path1):  # 一级文件夹里的子文件夹共分成四类
        path0 = os.path.join(path1, folder_name0)
        for folder_name1 in os.listdir(path0):  # 获取其中一个子文件夹内的子文件夹名
            path2 = os.path.join(path0, folder_name1)
            #print(path2)
            count = 1

            centers = []
            #centers = []
            for folder_name2 in os.listdir(path2):  # 获取子文件夹内的文件名
                folder_path = os.path.join(path2, folder_name2)
                #print(folder_path)
                img_path = folder_path


                EYE_AREA = 192
                EYE_AREA = int(EYE_AREA / 2)
                name0 = folder_name0
                name2 = folder_name1
                name = folder_name0 + '-' + folder_name1
                name_save = [name]
                name1 = folder_name2.split('.')[0]  # 不带后缀的文件名称
                eye_paramater = [name1]
                imgs_num = [count]

                calculatepuiple(centers)
            # centers.append([[654,654],[654,654]])
            centers = np.array(centers)
            left1x = 0
            lefty = 0
            right1x = 0
            righty = 0
            '''
            #平均数
            for i in range(len(centers)):
                left1x += centers[i, 0, 0]
                lefty += centers[i, 0, 1]
                right1x += centers[i, 1, 0]
                righty += centers[i, 1, 1]

            left2x = int(left1x/len(centers))
            left2y = int(lefty/len(centers))
            right2x = int(right1x/len(centers))
            right2y = int(righty/len(centers))
            '''

            #众数
            try:
                left2x = stats.mode(centers[:, 0, 0])
                left2y = stats.mode(centers[:, 0, 1])
                right2x = stats.mode(centers[:, 1, 0])
                right2y = stats.mode(centers[:, 1, 1])
                left2x = left2x[0]
                left2y = left2y[0]
                right2x = right2x[0]
                right2y = right2y[0]
            except Exception as e:
                left2x = 654
                left2y = 654
                right2x = 654
                right2y = 654
                print(e)
                print(path2)





            centers1 = []
            centers1.append(left2x)
            centers1.append(left2y)
            centers2 = []
            centers2.append(right2x)
            centers2.append(right2y)
            # centers1[0] = left2x
            # centers1[1] = left2y
            # centers2 = []
            # centers2[0] = right2x
            # centers2[1] = right2y

            centers3=[]
            centers3.append(centers1)
            centers3.append(centers2)
            centers3 = np.array(centers3)
            try:
                centers3 = centers3.squeeze(-1)  # 下降一维度
            #print(centers3)
            except Exception as e:
                centers3 = np.array([[654,654],[654,654]])
                print(e)
                print(path2)

            for folder_name2 in os.listdir(path2):  # 获取子文件夹内的文件名
                folder_path = os.path.join(path2, folder_name2)
                #print(folder_path)
                img_path = folder_path

                EYE_AREA = 192
                EYE_AREA = int(EYE_AREA / 2)
                name0 = folder_name0
                name2 = folder_name1
                name = folder_name0 + '-' + folder_name1
                name_save = [name]
                name1 = folder_name2.split('.')[0]  # 不带后缀的文件名称
                eye_paramater = [name1]
                imgs_num = [count]

                main_preprocess(centers3)
                count += 1

            #print(path2)