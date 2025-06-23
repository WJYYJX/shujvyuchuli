#导入所需的包
import numpy as np
import cv2

#导入npy文件路径位置
test = np.load(r'I:\jinzhou_4.5.10.11.16.17\train\image\1\x_jinzhou_0.npy')
img = test[:,:,5]
cv2.imshow("O", img)
cv2.waitKey(0)
print(test)