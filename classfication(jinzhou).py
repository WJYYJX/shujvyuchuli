import os
import shutil

# 读入分类的标签txt文件
#label_file = open("C:\\Users\\111\\Desktop\\list_label.txt", 'r')
label_file = open("G:\\goubangzi_left_labe.txt", 'r')
# 原始文件的根目录
#input_path = "C:\\Users\\111\\Desktop\\data_jinzhou\\筛选后\\宜州小学数据集\\宜州小学屈光度图像1（已去重）"
input_path = "F:\\goubangzi2000new\\left"
# 保存文件的根目录
output_path = "C:\\Users\\111\\Desktop\\goubangzi1\\左眼分类"
# 标签数组
#lables = ["-5", "-2.5", "-1", "-0.5","0", "0.5", "1", "2.5","5"]
lables = ["+0.00", "+0.25", "+0.50", "+0.75","+1.00", "+1.25", "+1.50", "+1.75","+2.00", "+2.25", "+2.50", "+2.75", "+3.00", "+3.25", "+3.50", "+3.75", "+4.00", "+4.25", "+4.50", "+4.75", "+5.00", "+5.25", "+5.50", "+6.25", "+7.00", "+7.25", "+7.50", "+7.75", "-0.25", "-0.50", "-0.75", "-1.00", "-1.25", "-1.50", "-1.75", "-2.00", "-2.25", "-2.50", "-2.75", "-3.00", "-3.25", "-3.50", "-3.75", "-4.00", "-4.25", "-4.50", "-4.75", "-5.00", "-5.25", "-5.50", "-5.75", "-6.00", "-6.25", "-6.50", "-6.75", "-7.00", "-7.25", "-7.50", "-7.75", "-8.00", "-8.25", "-8.50", "-8.75", "-9.00", "-9.25", "-9.50", "-9.75", "-10.25", "-10.50", "-11.50", "-13.50", "-15.00", "-15.50", "-20.75", "0.00", "0.25", "0.50", "0.75","1.00", "1.25", "1.50", "1.75","2.00", "2.25", "2.50", "2.75", "3.00", "3.25", "3.50", "3.75", "4.00", "4.25", "4.50", "4.75", "5.00", "5.25", "5.50", "6.50", "7.75", "8.00", "8.25", "9.75", "10.00", "10.25", "10.50", "14.00"]

# 一行行读入标签文件
data = label_file.readlines()
# 计数用
i = 1
# 遍历数据
for line in data:
    # 通过空格拆分成数组
    str1 = line.split('\t',1)
    # 第一个是文件名
    file_name = str1[0]
    # 第二个是标签类别，并去除最后的换行字符
    file_label = str1[1].strip()
    # 原始文件的路径
    old_file_path = os.path.join(input_path, file_name)
    if old_file_path!=None:

    # 新文件路径
        new_file_path = "C:\\Users\\111\\Desktop\\goubangzi1\\左眼分类\\"+file_label
        #new_file_path = "D:\\datasets\\xinjiangleft"

    # 如果文件名中有test字符，将其保存至test文件夹下的对应标签文件夹中
        if "test" in file_name:
            new_file_path = os.path.join(output_path, "test", lables[int(file_label) - 1])
    # 如果文件名中有 train 字符，将其保存至train文件夹下的对应标签文件夹中
        elif "train" in file_name:
            new_file_path = os.path.join(output_path, "train", lables[int(file_label) - 1])

    # 如果路径不存在，则创建
        if not os.path.exists(new_file_path):
            print("路径 " + new_file_path + " 不存在，正在创建......")
            os.makedirs(new_file_path)

    # 新文件位置
        new_file_path = os.path.join(new_file_path, file_name)
        print("" + str(i) + "\t正在将 " + old_file_path + " 复制到 " + new_file_path)
    # 复制文件
        try:
            #shutil.copyfile(old_file_path, new_file_path+'.zip')
            shutil.copytree(old_file_path , new_file_path,symlinks=False, ignore=None)
        except:
            print('fail',format(old_file_path))
            continue

        i = i + 1
# 完成提示
print("完成")
