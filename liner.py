import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'C:\Users\x\Desktop\OpenCV-Pic\4\img4.jpg', cv2.IMREAD_GRAYSCALE)  # GRAYSCALE

# 线性变换
a = 2
O = float(a) * img
O[O > 255] = 255  # 大于255要截断为255

# 数据类型的转换
O = np.round(O)
O = O.astype(np.uint8)

# 显示原图与变换后的图的效果
cv2.imshow("img", img)
cv2.imshow("O", O)
cv2.waitKey(0)
cv2.destroyAllWindows()
