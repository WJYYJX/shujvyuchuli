import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def FindBrightSpot(img, threshold_radio = 0.75):
    '''
    寻找角膜亮斑位置
    全局阈值分割后，计算每一块的连通域，连通域最大的两块质心的位置作为角膜亮斑位置
    :param threshold_radio:
    :param img: 图像
    :return: 角膜亮斑位置图标
    '''

    BrightSpotArea_max = 2000  # 角膜亮斑区域最大面积为1000
    BrightSpotArea_min = 1 # 最小面积

    # 高斯滤波+全局阈值
    blur = cv2.GaussianBlur(img, (5, 5), 0)  # 加了高斯模糊之后的效果还是很棒的
    max_blur = np.max(blur) #计算像素点最大灰度值
    ret, img_threshold = cv2.threshold(blur, int(max_blur * threshold_radio), 255, cv2.THRESH_BINARY) #最大灰度值的一定比例作为阈值

    # 计算连通域质心和面积
    nccomps, labels, stats, centroids = cv2.connectedComponentsWithStats(img_threshold)
    # https://blog.csdn.net/weixin_43552197/article/details/107445415博客里面有各个参数的解析
    # labels//和原图一样大的标记图
    # stats, //nccomps×5的矩阵 表示每个连通区域的外接矩形和面积（pixel）
    # centroids //nccomps×2的矩阵 表示每个连通区域的质心
    # connectivity //计算4连通还是8连通
    adjust_stats = stats.copy()
    m, n = stats.shape[:]
    ##！！！！角膜亮斑面积的筛选有待改进
    # 最大连通区域超过1000 或者小于1的去掉
    indexes_adjust_stats = [] #删除的索引
    for i in range(m):
        if not BrightSpotArea_min <= stats[i,4] <= BrightSpotArea_max:
            indexes_adjust_stats.append(i)
    stats = np.delete(stats, indexes_adjust_stats, axis=0)
    centers = np.delete(centroids, indexes_adjust_stats, axis=0)

    # 检测到的点太多，就把阈值调高一点
    m, n = stats.shape[:]
    if m >= 3:
        centers = FindBrightSpot(img, threshold_radio=threshold_radio+0.05)
    elif m <= 1:
        centers = FindBrightSpot(img, threshold_radio=threshold_radio - 0.05)
    #
    #或者取面积最大的两个点
    #stats_sort = stats[:, 4].argsort()  # 对连通区域的面积进行由小到大排序
    #stats = stats[stats_sort]
    #centroids = centroids[stats_sort]
    #centers = centroids[[-1, -2]]  # 取连通面积最大的两个点,就是角膜亮斑的位置

    centers = np.round(centers) #连通区域的质心是小数，对其进行圆整
    centers = centers.astype(np.uint16) # 转化为整数类型

    #确保第一个元素是左眼的位置
    if centers[0,0] > centers[1,0]:
        centers = centers[[1,0]]

    #确保两眼距离足够大
    if centers[1,0]-centers[0,0] <= 300 and m==2:
        raise Exception("检测的瞳孔位置不对")

    return centers

def FindBrightSpot_2(img, threshold_radio = 0.75):
    '''
    寻找角膜亮斑位置 版本2.0,不再使用迭代的算法，而是使用模板匹配的方法

    :param threshold_radio:
    :param img: 图像
    :return: 角膜亮斑位置图标
    '''

    BrightSpotArea_max = 2000  # 角膜亮斑区域最大面积为1000
    BrightSpotArea_min = 1 # 最小面积

    # 高斯滤波+全局阈值
    blur = cv2.GaussianBlur(img, (5, 5), 0)  # 加了高斯模糊之后的效果还是很棒的
    max_blur = np.max(blur)

    while 1:#死循环，确保检测出来两个以上的亮斑 #确保这两以上个亮斑的距离是合适的
        centers = FindMoreBrightSpots(blur, max_blur, threshold_radio, BrightSpotArea_min, BrightSpotArea_max)
        # 检测到的点太多，就把阈值调高一点
        m, n = centers.shape[:]
        if m >= 2: #如果有两个以上的亮斑就跳出循环
            #聚成两类，只有两类的中心点距离在合适范围内才跳出循环
            break
        else:
            threshold_radio -= 0.05

    centers = np.round(centers)  # 连通区域的质心是小数，对其进行圆整
    centers = centers.astype(np.int16)  # 转化为整数类型

    #如果有两个以上的亮斑，就两两组合，找出一个最佳的搭配
    briefs = np.array([[]]*3).T #用来存储亮斑两两组合的描述符
    if m > 2:
        for ii in range(m-1):
            for jj in range(m-1-ii):
                center0 = centers[ii]
                center1 = centers[ii+jj+1]
                if abs(center1[0]-center0[0])<200: # 如果两个匹配点距离很近就取消匹配 #!!!!又溢出了
                    continue
                else:
                    # 计算两点20*20邻域的相似性,使用基于像素差平方和(ssd)的描述符,这个比较简单，运算也比较快，也可以换成其它的描述符
                    # center1 第一个元素放的是x坐标(列数)，第二个元素放的是y坐标(行数)
                    ssd0 = np.sum(img[center0[1] - 10:center0[1] + 10, center0[0] - 10:center0[0] + 10])
                    ssd1 = np.sum(img[center1[1] - 10:center1[1] + 10, center1[0] - 10:center1[0] + 10])
                    brief = np.array([abs(int(ssd1) - int(ssd0)), ii, ii + jj + 1]) #靠，这里也会溢出
                    briefs = np.r_[briefs, [brief]]

        # 取描述符最小的两个匹配
        briefs_sort = briefs[:, 0].argsort()
        good_center0 = int(briefs[briefs_sort[0]][1])
        good_center1 = int(briefs[briefs_sort[0]][2])
        centers = centers[[good_center0, good_center1]]


    #确保第一个元素是左眼的位置
    if centers[0,0] > centers[1,0]:
        centers = centers[[1,0]]

    # 确保两眼距离足够大
    if centers[1, 0] - centers[0, 0] <= 300 and m == 2:
        raise Exception("检测的瞳孔位置不对")

    return centers
def FindBrightSpot_3(img1, hessianThreshold=2000):
    image1 = img1.copy()
    image1 = cv2.resize(image1, dsize=(800, 600))

    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)  # 经过观察关键点的响应发现，600确实是一个很合适的值，在锦州的数据集上，2000更合适
    keypoints1, descriptor1 = surf.detectAndCompute(image1, None)

    # 剔除size小于25的关键点
    keypoints1_choose = []
    key_response = []
    for keypoints in keypoints1:
        key_response.append(keypoints.response)
        if keypoints.size > 25:
            # 如果其角度是270度或者90度， 并且相应<3000说明这是来自边框上的点，需要先跳过
            if keypoints.response <= 3000 and (87 < keypoints.angle < 93 or 267 < keypoints.angle < 273):
                continue
            keypoints1_choose.append(keypoints)
    # print(key_response)

    # 大概计算出瞳孔的位置
    positions = np.array(
        [[keypoints.pt[0], keypoints.pt[1], keypoints.response] for keypoints in keypoints1_choose])

    if positions.ndim <= 1:
        print("没有检测出瞳孔")

        #raise Exception("检测的瞳孔位置不对")
    m, n = np.shape(positions)
    # 检测出的坐标点按照x坐标排序
    x_sort = positions[:, 0].argsort()
    positions = positions[x_sort]
    position_left = []
    position_right = []

    # 将坐标点划入左右两个集合里面
    left = positions[0, 0:2]
    right = positions[-1, 0:2]
    for ii in range(m):
        temp = positions[ii, 0:2]
        if np.linalg.norm(left - temp) < np.linalg.norm(right - temp):
            position_left.append(positions[ii])
        else:
            position_right.append(positions[ii])
    position_left = np.array(position_left)
    position_right = np.array(position_right)

    # 以response作为权重，去计算左右两个集合的中心
    m, n = np.shape(position_left)
    if m == 1:
        left_center = position_left[0, 0:2]
    else:
        response_sum = np.sum(position_left[:, 2])
        left_center = np.array([0, 0])
        for point in position_left:
            left_center = np.add(left_center, point[0:2] * point[2] / response_sum)
    left_center = left_center.reshape(2,)
    # left_center = np.mean(np.array(position_left), axis=0)
    left_center = np.round(left_center)
    m, n = np.shape(position_right)
    if m == 1:
        right_center = position_right[0, 0:2]
    else:
        response_sum = np.sum(position_right[:, 2])
        right_center = np.array([0, 0])
        for point in position_right:
            right_center = np.add(right_center, point[0:2] * point[2] / response_sum)
    right_center = right_center.reshape(2,)
    right_center = np.round(right_center)
    centers = np.array([left_center,right_center])
    centers = centers*2
    centers = centers.astype(np.int16) #转化为整数类型


    return centers

def FindMoreBrightSpots(blur, max_blur, threshold_radio, BrightSpotArea_min, BrightSpotArea_max):
    '''
    找出多个候选亮点
    :param blur: 经过模糊处理过后的图像的
    :param max_blur: 模糊后图像的最大灰度值
    :param threshold_radio: 阈值分割的比例
    :param BrightSpotArea_min: 亮点的最小值
    :param BrightSpotArea_max: 亮点的最大值
    :return:寻找出来的多个亮点
    '''
    ret, img_threshold = cv2.threshold(blur, int(max_blur * threshold_radio), 255, cv2.THRESH_BINARY)

    # 计算连通域质心和面积
    nccomps, labels, stats, centroids = cv2.connectedComponentsWithStats(img_threshold)
    # https://blog.csdn.net/weixin_43552197/article/details/107445415博客里面有各个参数的解析
    # labels//和原图一样大的标记图
    # stats, //nccomps×5的矩阵 表示每个连通区域的外接矩形和面积（pixel）
    # centroids //nccomps×2的矩阵 表示每个连通区域的质心
    # connectivity //计算4连通还是8连通
    adjust_stats = stats.copy()
    m, n = stats.shape[:]
    ##！！！！角膜亮斑面积的筛选有待改进
    # 最大连通区域超过1000 或者小于1的去掉
    indexes_adjust_stats = []  # 删除的索引
    for i in range(m):
        if not BrightSpotArea_min <= stats[i, 4] <= BrightSpotArea_max:
            indexes_adjust_stats.append(i)
    stats = np.delete(stats, indexes_adjust_stats, axis=0)
    centers = np.delete(centroids, indexes_adjust_stats, axis=0)
    return centers

def circleLeastFit( points):
    N =len(points)
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_y2 = 0
    sum_x3 = 0
    sum_y3 = 0
    sum_xy = 0
    sum_x1y2 = 0
    sum_x2y1 = 0

    for  i in range(N) :
#        print(points[i][0])
        x = float(points[i][0])
        y = float(points[i][1])
        x2 = x * x
        y2 = y * y
        sum_x += x
        sum_y += y
        sum_x2 += x2
        sum_y2 += y2
        sum_x3 += x2 * x
        sum_y3 += y2 * y
        sum_xy += x * y
        sum_x1y2 += x * y2
        sum_x2y1 += x2 * y
    C = N * sum_x2 - sum_x * sum_x
    D = N * sum_xy - sum_x * sum_y
    E = N * sum_x3 + N * sum_x1y2 - (sum_x2 + sum_y2) * sum_x
    G = N * sum_y2 - sum_y * sum_y
    H = N * sum_x2y1 + N * sum_y3 - (sum_x2 + sum_y2) * sum_y
    a = (H * D - E * G) / (C * G - D * D+1e-100)
    b = (H * C - E * D) / (D * D - G * C+1e-100)
    c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / N
    centerx = a / (-2)
    centery = b / (-2)
    rad = math.sqrt(a * a + b * b - 4 * c) / 2

    return np.array([centerx,centery,rad])
def PupilEdgeExtract(img_cut):
    '''
    从截取到的图片里提取瞳孔边缘
    :param img_cut:已经截取之后的左眼或者右眼图片
    :return:
    '''
    import cv2
    import numpy as np

    src = img_cut

    # 内部边缘
    img = cv2.GaussianBlur(src, (5, 5), 0)
    Canny_img_1 = cv2.Canny(img, 5, 15, (5, 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    Canny_img_1 = cv2.dilate(Canny_img_1, kernel)

    # 外部边缘
    equalize = cv2.equalizeHist(src)  # 直方图均衡化
    media_blur = cv2.medianBlur(equalize, 5)
    gaussian_blur = cv2.GaussianBlur(media_blur, (5, 5), 0)
    Canny_img_2_src = cv2.Canny(gaussian_blur, 90, 180, (11, 11))
    Canny_img_2 = cv2.dilate(Canny_img_2_src, kernel)

    # 综合内外边缘
    temp_result = cv2.multiply(Canny_img_1, Canny_img_2)
    result = cv2.multiply(temp_result, Canny_img_2_src)

    ###对比拟合圆、霍夫圆检测、轮廓检测之后发现，还是霍夫圆检测的效果比较好
    # 霍夫圆检测
    circles = cv2.HoughCircles(result, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=10, minRadius=10, maxRadius=100)#
    if circles is None: #如果霍夫圆没有拟合好，就使用自己的圆检测
        m, n = np.shape(result)
        points = [[j, i] for i in range(m) for j in range(n) if
                  result[i, j] >= 1]
        circles = circleLeastFit(points)
    circles = circles.reshape((-1, 3))
    sort_circles = circles[:, 2].argsort()  # cicrles 的第三列是圆的半径，在这里按照圆的半径由小到大排序
    circle = circles[sort_circles[-1]]  # 取圆的半径最大的那个圆

    circle_int = np.uint16(np.around(circle))
    '''plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(Canny_img_1,'gray')
    plt.subplot(1,3,2)
    plt.imshow(Canny_img_2, 'gray')
    plt.subplot(1,3,3)
    plt.imshow(result, 'gray')
    plt.show()'''

    return circle_int[0], circle_int[1], circle_int[2]

def PupilEdgeExtract_2(result,last_center):
    '''
    直接从已经提取边缘的瞳孔图像中进行圆拟合
    :param img_cut:已经截取之后的左眼或者右眼图片
    :return:
    '''
    import cv2
    import numpy as np

    ###对比拟合圆、霍夫圆检测、轮廓检测之后发现，还是霍夫圆检测的效果比较好
    # 霍夫圆检测
    circles = cv2.HoughCircles(result, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=10, minRadius=10, maxRadius=100)#
    if circles is None: #如果霍夫圆没有拟合好，就使用自己的圆检测 #霍夫圆检测都失效的话，自己的也不会有一个好的结果，直接使用原来的图片中心
        return last_center
        #m, n = np.shape(result)
        #points = [[j, i] for i in range(m) for j in range(n) if
         #         result[i, j] >= 1]
        #circles = circleLeastFit(points)
    circles = circles.reshape((-1, 3))
    sort_circles = circles[:, 2].argsort()  # cicrles 的第三列是圆的半径，在这里按照圆的半径由小到大排序
    circle = circles[sort_circles[-1]]  # 取圆的半径最大的那个圆

    circle_int = np.uint16(np.around(circle))
    return circle_int[0], circle_int[1], circle_int[2]

def ImgRemoveSpotAndJiemao(img_cut, x, y):
    '''
    移除图片中的亮斑和睫毛
    :param img_cut: 待移除睫毛和亮斑的图像
    :param x: 瞳孔拟合圆中心横坐标
    :param y: 瞳孔拟合圆中心纵坐标
    :return: 移除亮斑之后的图像
    '''

    # 消除亮斑

    img_remove = img_cut.copy()
    # 高斯滤波+全局阈值
    threshold_radio = 0.75
    blur = cv2.GaussianBlur(img_remove, (5, 5), 0)  # 加了高斯模糊之后的效果还是很棒的
    max_blur = np.max(blur)
    ret, img_threshold = cv2.threshold(blur, int(max_blur * threshold_radio), 255,
                                       cv2.THRESH_BINARY)
    # 膨胀操作 # 阈值分割出的区域还是比较小的，还需要再膨胀一下
    kernel = np.ones((3, 3), np.uint8)
    img_dilate = cv2.dilate(img_threshold, kernel, iterations=1)

    # 计算连通域 #通过计算连通域，求出亮斑区域的大小
    nccomps, labels, stats, centroids = cv2.connectedComponentsWithStats(img_dilate)

    # 亮斑上下几行像素的平均值 来代替亮斑的像素
    # 上下是几行，应当以亮斑范围的高度而定 #stats[1, 3]是像素高度
    RowNumber = int(np.ceil(stats[1, 3] / 2))
    for i in range(stats[1, 2]):  # stats[1,2]为联通区域的宽度
        UpRow = img_remove[stats[1, 1] - RowNumber:stats[1, 1], stats[1, 0] + i]
        DownRow = img_remove[stats[1, 1] + stats[1, 3]:stats[1, 1] + stats[1, 3] + RowNumber, stats[1, 0] + i]
        mean = int(np.round((np.mean(UpRow) + np.mean(DownRow)) / 2))
        img_remove[stats[1, 1]:stats[1, 1] + stats[1, 3], stats[1, 0] + i] = mean

    # 消除睫毛
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 6))  # 黑帽变换的结构元，
    blackhat = cv2.morphologyEx(img_remove, cv2.MORPH_BLACKHAT, kernel)  # 黑帽变换定位眉毛位置

    threshold_balckhat = 8  # 黑帽变换结果大于10的 被认定为睫毛位置
    img_jiemao_rectify = img_remove.copy()
    for ii in range(10):
        for jj in range(50):
            if blackhat[y - 5 + ii, x - 25 + jj] >= threshold_balckhat:
                # 把该点附近10行两列 的最大值作为该点的像素
                img_jiemao_rectify[y - 5 + ii, x - 25 + jj] = np.max(
                    img_remove[y - 5 + ii - 5: y - 5 + ii + 5, x - 25 + jj - 1:x - 25 + jj + 2])
    # !!!!!! img_cut与 img_jiemao_rectify 要考虑好#如果前后都用img_remove,前面修正好的值 会影响后面修正时的结果。

    return img_jiemao_rectify

def calculate(image1, image2):
    # 灰度直方图算法
    # 计算单通道的直方图的相似值
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


def pHash(img):
    # 感知哈希算法
    # 缩放32*32
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC

    # 转换为灰度图
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(img))
    # opencv实现的掩码操作
    dct_roi = dct[0:8, 0:8]

    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash

def cmpHash(hash1, hash2):
    # Hash值对比
    # 算法中1和0顺序组合起来的即是图片的指纹hash。顺序不固定，但是比较的时候必须是相同的顺序。
    # 对比两幅图的指纹，计算汉明距离，即两个64位的hash值有多少是不一样的，不同的位数越小，图片越相似
    # 汉明距离：一组二进制数据变成另一组数据所需要的步骤，可以衡量两图的差异，汉明距离越小，则相似度越高。汉明距离为0，即两张图片完全一样
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n




