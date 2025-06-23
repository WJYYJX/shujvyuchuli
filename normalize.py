# 直方图正规化API
# 灰度级主要在0~150之间，造成图像对比度较低，可用直方图正规化将图像灰度级拉伸到0~255,使其更清晰
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 灰度图像转化为ndarray类型
if __name__ == "__main__":
    src = cv2.imread(r'C:\Users\111\Desktop\pridict\segment-32.jpg(1).jpg', cv2.IMREAD_ANYCOLOR)
    dst = np.zeros_like(src)
    cv2.normalize(src, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # 公式
    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 计算灰度直方图
    grayHist = cv2.calcHist([src], [0], None, [256], [0, 256])
    grayHist1 = cv2.calcHist([dst], [0], None, [256], [0, 256])
    # 画出直方图
    x_range = range(256)
    plt.plot(x_range, grayHist, 'r', linewidth=1.5, c='black')
    plt.plot(x_range, grayHist1, 'r', linewidth=1.5, c='b')
    # 设置坐标轴的范围
    y_maxValue = np.max(grayHist)
    plt.axis([0, 255, 0, y_maxValue])  # 画图范围
    plt.xlabel("gray Level")
    plt.ylabel("number of pixels")
    plt.show()
