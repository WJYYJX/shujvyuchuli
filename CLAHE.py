# 限制对比度的自适应直方图均衡化
import cv2

src = cv2.imread(r'C:\Users\111\Desktop\pridict\segment-32.jpg(1).jpg', cv2.IMREAD_ANYCOLOR)
# 创建CLAHE对象 clipLimit限制对比度，tileGridSize块的大小
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# 限制对比度的自适应阈值
dst = clahe.apply(src)

cv2.namedWindow("src", cv2.WINDOW_NORMAL)
cv2.namedWindow("clahe", cv2.WINDOW_NORMAL)
cv2.imshow("src", src)
cv2.imshow("clahe", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
