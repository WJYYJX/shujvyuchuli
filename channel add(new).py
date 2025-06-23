import cv2
import os
import shutil
import numpy as np

if __name__ == '__main__':
    train_image = []
    train_label = []
    path1 = r"C:\Users\111\Desktop\xinjiang_select\train"  # 包含四类的文件夹路径 C:\Users\111\Desktop\zhongzhuan\val

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
                label = folder_name1
                label = np.array(label)
                # img1 = cv2.imread(path3 + '/1.bmp', cv2.COLOR_BGR2RGB)
                # img2 = cv2.imread(path3 + '/2.bmp', cv2.COLOR_BGR2RGB)
                # img3 = cv2.imread(path3 + '/3.bmp', cv2.COLOR_BGR2RGB)
                # img4 = cv2.imread(path3 + '/4.bmp', cv2.COLOR_BGR2RGB)
                # img5 = cv2.imread(path3 + '/5.bmp', cv2.COLOR_BGR2RGB)
                # img6 = cv2.imread(path3 + '/6.bmp', cv2.COLOR_BGR2RGB)
                # img7 = cv2.imread(path3 + '/7.bmp', cv2.COLOR_BGR2RGB)
                # img8 = cv2.imread(path3 + '/8.bmp', cv2.COLOR_BGR2RGB)
                # img9 = cv2.imread(path3 + '/9.bmp', cv2.COLOR_BGR2RGB)
                # img10 = cv2.imread(path3 + '/10.bmp', cv2.COLOR_BGR2RGB)
                # img11 = cv2.imread(path3 + '/11.bmp', cv2.COLOR_BGR2RGB)
                # img12 = cv2.imread(path3 + '/12.bmp', cv2.COLOR_BGR2RGB)
                try:
                    img1 = cv2.imread(path3 + '/1.bmp', cv2.COLOR_BGR2RGB)
                    img2 = cv2.imread(path3 + '/2.bmp', cv2.COLOR_BGR2RGB)
                    #img3 = cv2.imread(path3 + '/3.bmp', cv2.COLOR_BGR2RGB)
                    img4 = cv2.imread(path3 + '/4.bmp', cv2.COLOR_BGR2RGB)
                    # img5 = cv2.imread(path3 + '/5.bmp', cv2.COLOR_BGR2RGB)
                    img6 = cv2.imread(path3 + '/6.bmp', cv2.COLOR_BGR2RGB)
                    # img7 = cv2.imread(path3 + '/7.bmp', cv2.COLOR_BGR2RGB)
                    # img8 = cv2.imread(path3 + '/8.bmp', cv2.COLOR_BGR2RGB)
                    # img9 = cv2.imread(path3 + '/9.bmp', cv2.COLOR_BGR2RGB)
                    img10 = cv2.imread(path3 + '/10.bmp', cv2.COLOR_BGR2RGB)
                    # img11 = cv2.imread(path3 + '/11.bmp', cv2.COLOR_BGR2RGB)
                    # img12 = cv2.imread(path3 + '/12.bmp', cv2.COLOR_BGR2RGB)
                    # img13 = cv2.imread(path3 + '/13.bmp', cv2.COLOR_BGR2RGB)
                    # img14 = cv2.imread(path3 + '/14.bmp', cv2.COLOR_BGR2RGB)
                    # img15 = cv2.imread(path3 + '/15.bmp', cv2.COLOR_BGR2RGB)
                    img16 = cv2.imread(path3 + '/16.bmp', cv2.COLOR_BGR2RGB)
                    # img17 = cv2.imread(path3 + '/17.bmp', cv2.COLOR_BGR2RGB)
                    # img18 = cv2.imread(path3 + '/18.bmp', cv2.COLOR_BGR2RGB)
                    # img19 = cv2.imread(path3 + '/19.bmp', cv2.COLOR_BGR2RGB)
                    # img20 = cv2.imread(path3 + '/20.bmp', cv2.COLOR_BGR2RGB)
                    # img21 = cv2.imread(path3 + '/21.bmp', cv2.COLOR_BGR2RGB)
                    # img22 = cv2.imread(path3 + '/22.bmp', cv2.COLOR_BGR2RGB)
                except FileNotFoundError:
                    print(path3+"没有图片 ")

                savepath = r"C:\xinjiang_select_rgb\train"

                savepath1 = savepath + '/' + 'image' '/' + folder_name1 + '/'
                folder = os.path.exists(savepath1)
                if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
                    os.makedirs(savepath1)  # makedirs 创建文件时如果路径不存在会创建这个路径

                savepath2 = savepath + '/' + 'label' '/' + folder_name1 + '/'
                folder1 = os.path.exists(savepath2)
                if not folder1:  # 判断是否存在文件夹如果不存在则创建为文件夹
                    os.makedirs(savepath2)  # makedirs 创建文件时如果路径不存在会创建这个路径

                #res = cv2.merge([img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12])
                try:
                    res = cv2.merge([img1, img2, img4, img6, img10, img16])
                    res = np.array(res)
                    if res.shape == (192, 192, 6):
                        np.save(savepath1 + folder_name2 + '.npy', res)
                        np.save(savepath2+  folder_name2 + '.npy', label)

                except Exception as e:
                    print(e)
                    print(path3)




                # cv2.imwrite('cat.png',res)

                #print(train_label)




