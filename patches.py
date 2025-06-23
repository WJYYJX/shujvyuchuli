import cv2
import numpy as np
import os


def split_by_blocksize_cv(image_path, output_dir, block_w, block_h):
    """使用OpenCV按块尺寸切割"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    h, w = img.shape[:2]
    os.makedirs(output_dir, exist_ok=True)

    index = 0
    for y in range(0, h, block_h):
        for x in range(0, w, block_w):
            # 计算切割区域
            y_end = min(y + block_h, h)
            x_end = min(x + block_w, w)

            # 切割并保存
            block = img[y:y_end, x:x_end]
            cv2.imwrite(f"{output_dir}/block_{index}.jpg", block)
            index += 1


# 示例：切割为256x256块
split_by_blocksize_cv("C:\\Users\\111\\Desktop\\pupil\\patch.jpg", "C:\\Users\\111\\Desktop\\pupil", 32, 32)