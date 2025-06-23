import cv2
import os
import numpy as np




# 灰度图像转化为ndarray类型
I = cv2.imread(r'C:/Users/111/Desktop/pupil/-2.5/img8.jpg', cv2.IMREAD_ANYCOLOR)
# 图像归一化
fI = I / 255.0
# 伽马变化
gamma = 0.4
O = np.power(fI, gamma)

# 显示伽马变化后的结果
    #cv2.namedWindow("I", cv2.WINDOW_NORMAL)
cv2.namedWindow("O", cv2.WINDOW_NORMAL)
cv2.imshow("I", I)
cv2.imshow("O", O)
cv2.waitKey(0)
cv2.destroyAllWindows()
