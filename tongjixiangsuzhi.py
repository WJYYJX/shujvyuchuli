from PIL import Image
import numpy as np


def analyze_pixel_distribution(tif_path):
    # 打开TIFF文件
    with Image.open(tif_path) as img:
        total_counts = {}
        # 处理多帧图像（如动图或多页TIFF）
        for frame in range(img.n_frames):
            try:
                img.seek(frame)  # 跳转到当前帧
                pixels = np.array(img)  # 转换为numpy数组
                # 统计当前帧的像素分布
                values, counts = np.unique(pixels, return_counts=True)
                # 合并统计结果
                for value, count in zip(values, counts):
                    total_counts[value] = total_counts.get(value, 0) + count
            except EOFError:
                # 已处理所有帧
                break
        return total_counts


if __name__ == "__main__":
    tif_path = r'C:\Users\111\Documents\WeChat Files\wxid_vejx44n1lkj421\FileStorage\File\2025-04\H huidu.tif'  # 替换为你的文件路径
    distribution = analyze_pixel_distribution(tif_path)

    # 按像素值排序后打印结果
    print("Pixel Value : Count")
    for value in sorted(distribution.keys()):
        print(f"{value:10} : {distribution[value]}")