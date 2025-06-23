from itertools import repeat
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

NUM_THREADS = os.cpu_count()


def calc_channel_sum(img_path):  # 计算均值的辅助函数，统计单张图像颜色通道和，以及像素数量
    img = np.array(np.load(img_path, allow_pickle=True)) / 255.0  # 准换为RGB的array形式

    h, w, _ = img.shape
    pixel_num = h * w
    channel_sum = img.sum(axis=(0, 1))  # 各颜色通道像素求和
    return channel_sum, pixel_num


def calc_channel_var(img_path, mean):  # 计算标准差的辅助函数
    img = np.array(np.load(img_path, allow_pickle=True)) / 255.0
    channel_var = np.sum((img - mean) ** 2, axis=(0, 1))
    return channel_var

def mean_and_var(data_path,data_format='*.bmp',decimal_places=4):
    """
    计算均值方差
    @param data_path: 数据集路径
    @param data_format: 图片格式（默认为png）
    @param decimal_places: 均值和方差，保留的小数位数（默认为4）
    @return:
    """
    print("Data root is ",data_path)
    train_path =[]
    for folder_name0 in os.listdir(data_path):  # 一级文件夹里的子文件夹共分成四类
        path0 = os.path.join(data_path, folder_name0)
        for folder_name1 in os.listdir(path0):  # 获取其中一个子文件夹内的子文件夹名
            path2 = os.path.join(path0, folder_name1)
            #print(path2)
            for folder_name2 in os.listdir(path2):  # 获取其中一个子文件夹内的子文件夹名
                path3 = os.path.join(path2, folder_name2)
                train_path.append(path3)

    img_f = list(train_path)
    n = len(img_f)
    print(f'Data Nums is : {n}')
    print("Calculate the mean value")
    result = ThreadPool(NUM_THREADS).imap(calc_channel_sum, img_f)  # 多线程计算
    channel_sum = np.zeros(6)
    cnt = 0
    pbar = tqdm(enumerate(result), total=n)
    for i, x in pbar:
        channel_sum += x[0]
        cnt += x[1]
    mean = channel_sum / cnt

    mean=np.around(mean, decimal_places)  # 使用around()函数保留小数位数
    print('R_mean,G_mean,B_mean is ',mean)

    print("Calculate the var value")
    result = ThreadPool(NUM_THREADS).imap(lambda x: calc_channel_var(*x), zip(img_f, repeat(mean)))
    channel_sum = np.zeros(6)
    pbar = tqdm(enumerate(result), total=n)
    for i, x in pbar:
        channel_sum += x
    var = np.sqrt(channel_sum / cnt)
    var = np.around(var, decimal_places)  # 使用around()函数保留小数位数
    print('R_var,G_var,B_var is ', var)


if __name__ == '__main__':
    mean_and_var('F:/moniyan_1.2.4.6.10.16/image')#F:\moniyandingzhidata\shujvji\val\0.50\POS X 0 Y 0 Z 8
