import cv2
import os
import shutil
import numpy as np

if __name__ == '__main__':
    train_image = []
    train_label = []
    path1 = r"C:\Users\111\Desktop\zhongzhuan\xinjiang-huafen\val"  # 包含四类的文件夹路径 C:\Users\111\Desktop\zhongzhuan\val

    for folder_name0 in os.listdir(path1):  # 一级文件夹里的子文件夹共分成四类
        path0 = os.path.join(path1, folder_name0)
        for folder_name1 in os.listdir(path0):  # 获取其中一个子文件夹内的子文件夹名
            path2 = os.path.join(path0, folder_name1)
            #print(path2)
            for folder_name2 in os.listdir(path2):  # 获取其中一个子文件夹内的子文件夹名
                path3 = os.path.join(path2, folder_name2)

                try:
                    img_name_list = os.listdir(path3)  #
                except FileNotFoundError:
                    print("没有该学生文件夹  ")

                source_file = path3 + '/1.bmp'
                savepath = r"F:\moniyan_channelclass"

                savepath0 = savepath + '/' +folder_name0 + '/' + '/' + 'channl0'

                savepath1 = savepath0 + '/' + folder_name1 + '-' + folder_name2 + '-1'+ '.bmp'
                folder = os.path.exists(savepath0)
                if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
                    os.makedirs(savepath0)  # makedirs 创建文件时如果路径不存在会创建这个路径
                shutil.copy(source_file, savepath1)
                #












