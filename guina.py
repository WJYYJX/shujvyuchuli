import os
import cv2
import shutil
if __name__ == '__main__':
    path1 = "F:/test"  # 包含四类的文件夹路径#"C:/Users/111/Desktop/rank/"
    for folder_name0 in os.listdir(path1):  # 一级文件夹里的子文件夹共分成四类
        path0 = os.path.join(path1, folder_name0)
        for folder_name1 in os.listdir(path0):  # 获取其中一个子文件夹内的子文件夹名
            path2 = os.path.join(path0, folder_name1)
            print(path2)
            for folder_name2 in os.listdir(path2):  # 获取子文件夹内的文件名
                path3 = os.path.join(path2, folder_name2)
                lst = path3.split(" ")
                lst1 = lst[-1].split("_")
                lst2 = lst1[-1].split(".")
                lst3 = lst[-1].split("_")

                source_file = path3
                savepath = r"G:\moniyan_translate"

                savepath0 = savepath + '/' + lst2[0] + '/' + folder_name0 + '/' + lst[1] +' ' + lst[2] +' ' + lst[3] +' ' + lst[4] +' ' + lst[5] +' ' + lst3[0]

                savepath1 = savepath0 + '/' + lst1[1] + '.png'
                folder = os.path.exists(savepath0)
                if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
                    os.makedirs(savepath0)  # makedirs 创建文件时如果路径不存在会创建这个路径
                try:
                    shutil.copy(source_file, savepath1)
                except Exception as e:
                    print(e)
                    print(path3)