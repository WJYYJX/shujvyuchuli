import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
# 设置中文字体和符号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统可用
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def process_bright_spots(image_path):
    # 读取图像并保持原始位深
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("无法读取图像，请检查路径")

    # 转换位深并创建彩色图像用于标记
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    # 根据位深确定最大值并二值化
    max_val = 255 if img.dtype == np.uint8 else 65535
    _, thresh = cv2.threshold(img, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # 过滤小轮廓（可根据实际调整面积阈值）
    min_area = 10
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # 存储强度值和坐标
    intensities = []
    coordinates = []

    # 处理每个亮斑
    for c in filtered_contours:
        # 创建掩膜
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

        # 计算平均强度
        mean_val = cv2.mean(img, mask=mask)[0]
        intensities.append(mean_val)
        cv2.drawContours(img_color, [c], -1, (0, 255, 0), 1)  # 绿色轮廓
        # 计算中心点并记录
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            coordinates.append((cx, cy))

    # 绘制标记（根据位深调整颜色）
    marker_color = (0, 255, 0) if img.dtype == np.uint8 else (0, 65535, 0)
    for (cx, cy) in coordinates:
        cv2.circle(img_color, (cx, cy), 1, marker_color, -1)

    # 保存处理后的图像
    cv2.imwrite("processed_image.jpg", img_color)

    # 统计强度分布
    rounded = np.round(intensities).astype(int)
    unique, counts = np.unique(rounded, return_counts=True)

    # 生成直方图
    plt.bar(unique, counts, width=0.8)
    plt.xlabel("强度值")
    plt.ylabel("亮斑数量")
    plt.title("亮斑强度分布")
    plt.savefig("intensity_histogram.png", dpi=150)
    plt.close()

    # 保存CSV文件
    with open("intensity_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["强度", "数量"])
        for u, c in zip(unique, counts):
            writer.writerow([u, c])


if __name__ == "__main__":
    process_bright_spots(r'C:\Users\111\Documents\WeChat Files\wxid_vejx44n1lkj421\FileStorage\File\2025-04\H2 huidu.tif')
    print("处理完成，已生成：processed_image.tif, intensity_histogram.png, intensity_stats.csv")