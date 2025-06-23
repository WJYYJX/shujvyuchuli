import numpy as np
import os
import random
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import keras
name = 'zhongzhuan'
path = 'C:/Users/111/Desktop/' + name
def get_normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1993, 0.2010], to_Tensor=True):
    def normalizer(input_image):
        if to_Tensor:
            image = input_image / 255
        else:
            image = input_image

        reshape_shape = (1, 1, 3) if keras.backend.image_data_format() == 'channels_last' else (3, 1, 1)
        shaped_mean = np.reshape(mean, reshape_shape)
        shaped_std = np.reshape(std, reshape_shape)

        image = (image - shaped_mean) / shaped_std
        return image
    return normalizer
def image_label(imageLabel, label2idx, i):
    """
    返回图片的label
    """
    if imageLabel not in label2idx:
        lb = imageLabel.split('zhongzhuan\\')[1]
        label2idx[imageLabel] = lb
        i = i + 1
    # 返回的是字典类型
    return label2idx, i


def image2npy(dir_path=path, testScale=0.2):
    """
    生成npy文件
    """
    i = 0
    label2idx = {}
    data = []
    for root, dirs, files in os.walk(dir_path):
        # print(root, dirs, files)
        for Ufile in tqdm(files):
            # Ufile是文件名
            img_path = os.path.join(root, Ufile)  # 文件的所在路径
            File = root.split('/')[-1]  # 文件所在文件夹的名字, 也就是label

            # 读取image和label数据
            img_data = cv2.imread(img_path)
            img_data = cv2.resize(img_data, (128, 128))  # 调整图像大小
            label2idx, i = image_label(File, label2idx, i)
            label = label2idx[File]  # 生成图像的label

            # 存储image和label数据
            # normal = get_normalize()  # 归一化处理数据
            data.append([np.array(img_data), label])
    # 随机打乱数据
    random.shuffle(data)

    # 训练集和测试集的划分
    # testNum = int(len(data) * testScale)
    testNum = 200
    train_data = data[testNum:]  # 训练集
    test_data = data[:testNum]  # 测试集

    # 测试集的输入输出和训练集的输入输出
    X_train = np.array([i[0] for i in train_data])  # 训练集特征
    y_train = np.array([i[1] for i in train_data])  # 训练集标签
    y_train = y_train.reshape(-1, 1)  # -1表示任意行数，1表示1列
    y_train = y_train.astype(np.float)
    X_test = np.array([i[0] for i in test_data])  # 测试集特征
    y_test = np.array([i[1] for i in test_data])  # 测试集标签
    y_test = y_test.reshape(-1, 1)  # -1表示任意行数，1表示1列
    y_test = y_test.astype(np.float)
    print(len(X_train), len(y_train), len(X_test), len(y_test))

    # 保存文件
    # 判断目录是否存在，不存在就创建
    dirs = os.path.join('C:/Users/111/Desktop/npy_data', name)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.save(os.path.join(dirs, 'x_train'), X_train)
    np.save(os.path.join(dirs, 'y_train'), y_train)
    np.save(os.path.join(dirs, 'x_test'), X_test)
    np.save(os.path.join(dirs, 'y_test'), y_test)
    # np.savez_compressed(os.path.join(dirs, 'x_norm'), X_train[::10])

    return label2idx

label2idx = image2npy(dir_path=path, testScale=0.2)
print(label2idx)

# 随机检查label与图片是否可以对应上
# 从train中抽取9个image和9个label
image_no = np.random.randint(0, 100, size=9)  # 随机挑选9个数字

train_images = np.load(os.path.join('C:/Users/111/Desktop/npy_data/'+name, 'x_train.npy'))
train_labels = np.load(os.path.join('C:/Users/111/Desktop/npy_data/'+name, 'y_train.npy'))
fig, axes = plt.subplots(nrows=3, ncols=3,figsize=(7,7))

for i in range(3):
    for j in range(3):
        axes[i][j].imshow(train_images[image_no[i*3+j]])
        axes[i][j].set_title(train_labels[image_no[i*3+j]])
plt.tight_layout()