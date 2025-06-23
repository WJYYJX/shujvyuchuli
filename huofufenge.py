import cv2
import numpy as np
import os # 导入 os 模块用于路径检查

def circleLeastFit(points):
    '''
    使用最小二乘法拟合圆。
    当霍夫圆检测失败时作为备用方案。
    :param points: 边缘点的列表，每个点为 [x, y]
    :return: 拟合圆的 (中心x, 中心y, 半径)
    '''
    if len(points) < 3:
        # 如果点少于3个，无法拟合圆，返回默认值
        print("警告: 拟合圆的点不足3个，返回默认圆。")
        return np.array([0, 0, 0])

    points = np.array(points, dtype=np.float32)
    x = points[:, 0]
    y = points[:, 1]

    # 构建线性方程组 Ax = B 来求解圆的参数 (a, b, c)
    # 圆方程形式: x^2 + y^2 + ax + by + c = 0
    N = len(points)
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sxx = np.sum(x**2)
    Syy = np.sum(y**2)
    Sxy = np.sum(x * y)
    Sxxx = np.sum(x**3)
    Sxyy = np.sum(x * y**2)
    Sxxy = np.sum(x**2 * y)
    Syyy = np.sum(y**3)

    A = np.array([
        [Sxx, Sxy, Sx],
        [Sxy, Syy, Sy],
        [Sx, Sy, N]
    ])

    B = np.array([
        [-Sxxx - Sxyy],
        [-Sxxy - Syyy],
        [-Sxx - Syy]
    ])

    try:
        # 求解线性方程组
        sol = np.linalg.solve(A, B)
        a, b, c = sol.flatten()

        # 根据参数计算圆心和半径
        xc = -a / 2
        yc = -b / 2
        R = np.sqrt(xc**2 + yc**2 - c)

        return np.array([xc, yc, R])
    except np.linalg.LinAlgError:
        # 如果线性系统无法求解，打印警告并返回默认值
        print("警告: 无法求解圆拟合的线性系统。返回默认圆。")
        return np.array([0, 0, 0])

def PupilEdgeExtract(img_cut):
    '''
    从截取到的图片里提取瞳孔边缘
    :param img_cut: 已经截取之后的左眼或者右眼图片 (灰度图)
    :return: 瞳孔圆的 (中心x, 中心y, 半径), 以及作为霍夫圆检测输入的边缘图像
    '''
    src = img_cut

    # --- 内部边缘检测 ---
    # 对图像进行高斯模糊以减少噪声。增大核大小（例如从5到7）可以更平滑图像，减少噪声。
    img = cv2.GaussianBlur(src, (7, 7), 0) # 示例：将核大小从 (5,5) 增大到 (7,7)
    # 使用Canny算法检测边缘。提高阈值（例如从5,15到10,30）可以减少弱边缘，使得边缘更“干净”。
    Canny_img_1 = cv2.Canny(img, 10, 30, (5, 5)) # 示例：将阈值从 5,15 调整到 10,30
    # 定义矩形结构元素用于形态学操作。减小核大小（例如从5,5到3,3）可以减少膨胀程度。
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # 示例：将核大小从 (5,5) 减小到 (3,3)
    # 对边缘进行膨胀操作，连接断裂的边缘
    Canny_img_1 = cv2.dilate(Canny_img_1, kernel)

    # --- 外部边缘检测 ---
    # 对图像进行直方图均衡化以增强对比度
    equalize = cv2.equalizeHist(src)
    # 对均衡化后的图像进行中值模糊以进一步降噪。增大核大小可以减少噪声。
    media_blur = cv2.medianBlur(equalize, 7) # 示例：将核大小从 5 增大到 7
    # 再次进行高斯模糊。增大核大小可以减少噪声。
    gaussian_blur = cv2.GaussianBlur(media_blur, (7, 7), 0) # 示例：将核大小从 (5,5) 增大到 (7,7)
    # 使用Canny算法检测外部边缘。提高阈值（例如从90,180到120,240）可以减少弱边缘。
    Canny_img_2_src = cv2.Canny(gaussian_blur, 120, 240, (11, 11)) # 示例：将阈值从 90,180 调整到 120,240
    # 对外部边缘进行膨胀
    Canny_img_2 = cv2.dilate(Canny_img_2_src, kernel)

    # --- 综合内外边缘 ---
    # 将内部边缘和外部边缘相乘（逻辑与操作），以找到共同的、更强的边缘
    temp_result = cv2.multiply(Canny_img_1, Canny_img_2)
    # 再次与原始的外部边缘相乘，进一步细化结果
    result = cv2.multiply(temp_result, Canny_img_2_src)

    ### 霍夫圆检测、轮廓检测、对比拟合圆之后，霍夫圆检测的效果通常较好。
    # --- 霍夫圆检测 ---
    # 使用霍夫梯度法检测圆
    # 参数说明:
    # result: 8位灰度输入图像
    # cv2.HOUGH_GRADIENT: 检测方法
    # 1: dp - 图像分辨率的倒数，dp=1表示与输入图像相同的分辨率
    # 100: minDist - 两个圆心之间的最小距离。增大此值可以避免检测到过于靠近的圆。
    # param1: Canny边缘检测的高阈值。此参数会影响霍夫空间中圆的累加器阈值。
    # param2: 累加器阈值。越小检测到的假圆越多，越大越严格。减小此值可能检测到更多圆，增大则更少。
    # minRadius: 最小圆半径
    # maxRadius: 最大圆半径
    circles = cv2.HoughCircles(result, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=10, minRadius=10, maxRadius=100)

    if circles is None:
        # 如果霍夫圆检测没有找到圆，则使用自定义的最小二乘法圆拟合
        m, n = np.shape(result)
        # 提取所有非零像素点（即边缘点）
        points = [[j, i] for i in range(m) for j in range(n) if result[i, j] > 0]
        # 确保有足够的点进行拟合
        if len(points) < 3:
            print("警告: 没有足够的边缘点进行圆拟合。返回默认圆。")
            return 0, 0, 0, result # 返回一个默认的虚拟圆和边缘图像

        circle_data = circleLeastFit(points)
        circle = circle_data # circleLeastFit 返回的是 (x, y, r) 格式
    else:
        # 如果霍夫圆检测到圆，则选择半径最大的那个圆
        circles = circles.reshape((-1, 3))
        # 按照半径大小进行排序 (第三列是半径)
        sort_circles = circles[:, 2].argsort()
        # 取半径最大的那个圆
        circle = circles[sort_circles[-1]]

    # 将圆的参数四舍五入并转换为无符号16位整数
    circle_int = np.uint16(np.around(circle))

    # 返回圆心坐标、半径和作为霍夫圆检测输入的边缘图像
    return circle_int[0], circle_int[1], circle_int[2], result

# --- 程序主入口，用于演示 ---
if __name__ == "__main__":
    # 直接指定图片路径，不再需要用户输入
    img_path = "C:/Users/111/Desktop/2222.png"

    # 检查文件是否存在
    if not os.path.exists(img_path):
        print(f"错误: 文件 '{img_path}' 不存在。请检查路径是否正确。")
        exit()

    # 以灰度模式读取图片
    input_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 检查图片是否成功加载
    if input_img is None:
        print(f"错误: 无法从 '{img_path}' 加载图片。请确保文件是有效的图片格式。")
        exit()

    print("开始提取瞳孔边缘...")
    # 调用瞳孔边缘提取函数，现在它会返回第四个值：霍夫圆的输入边缘图像
    center_x, center_y, radius, hough_input_edges = PupilEdgeExtract(input_img)
    print(f"提取到的瞳孔圆心: ({center_x}, {center_y}), 半径: {radius}")

    # 为了可视化效果，将灰度图像转换为BGR格式，以便绘制彩色圆圈
    display_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    # 在图像上绘制检测到的瞳孔圆 (绿色，粗细为2)
    cv2.circle(display_img, (center_x, center_y), radius, (0, 255, 0), 2)
    # 在圆心绘制一个红色小点
    cv2.circle(display_img, (center_x, center_y), 2, (0, 0, 255), -1)

    # 保存结果图像到文件
    output_filename = 'pupil_detection_result.png'
    cv2.imwrite(output_filename, display_img)
    print(f"结果图片已保存为: {output_filename}")

    # 保存霍夫圆检测的输入边缘图像
    # 将边缘图像转换为BGR格式以便保存为彩色图片 (可选，但通常更易于查看)
    hough_input_edges_display = cv2.cvtColor(hough_input_edges, cv2.COLOR_GRAY2BGR)
    hough_input_edges_filename = 'hough_input_edges.png'
    cv2.imwrite(hough_input_edges_filename, hough_input_edges_display)
    print(f"霍夫圆检测的输入边缘图像已保存为: {hough_input_edges_filename}")
    print("请检查当前目录下的 'pupil_detection_result.png' 和 'hough_input_edges.png' 文件以查看效果。")
