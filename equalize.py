import cv2
import numpy as np
import math

if __name__ == "__main__":
    I = cv2.imread(r'C:\Users\111\Desktop\pridict\segment-32.jpg(1).jpg', cv2.IMREAD_GRAYSCALE)

    O = cv2.equalizeHist(I)

    cv2.namedWindow("I", cv2.WINDOW_NORMAL)
    cv2.namedWindow("O", cv2.WINDOW_NORMAL)
    cv2.imshow("I", I)
    cv2.imshow("O", O)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
