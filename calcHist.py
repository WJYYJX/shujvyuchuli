from matplotlib import pyplot as plt
import cv2

img = cv2.imread(r'F:/data/moniyan/train/10D/four/14/segment-31.jpg.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap=plt.cm.gray)

# images，必须用方括号括起来。
# channels，是用于计算直方图的通道，这里使用灰度图计算直方图，所以就直接使用第一个通道。
# Mask，图像掩模，没有使用则填写None。
# histSize，表示这个直方图分成多少份（即多少个直方柱）。
# ranges，表示直方图中各个像素的值，[0.0, 256.0]表示直方图能表示像素值从0.0到256的像素。
# accumulate，为一个布尔值，用来表示直方图是否叠加。

hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("gray Level")
plt.ylabel("number of pixels")

plt.plot(hist)
plt.xlim([0, 256])
plt.show()
