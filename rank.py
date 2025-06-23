import os

if __name__ == '__main__':
    path1 = "G:/goubangzi——origin"  # 包含四类的文件夹路径#"C:/Users/111/Desktop/rank/"
    for folder_name0 in os.listdir(path1):  # 一级文件夹里的子文件夹共分成四类
        path0 = os.path.join(path1, folder_name0)
        for folder_name1 in os.listdir(path0):  # 获取其中一个子文件夹内的子文件夹名
            path2 = os.path.join(path0, folder_name1)
            print(path2)
            count = 1
            files = os.listdir(path2)
            files.sort(key=lambda x: int(x.split('.')[0]))
            for folder_name2 in files:  # 获取子文件夹内的文件名
                folder_path = os.path.join(path2, folder_name2)
                print(folder_path)
                name1 = folder_name1.split('.')[0]  # 不带后缀的文件名称
                print(name1)
                if os.path.isfile(folder_path):  # 判断文件是否存在
                    new_name = "{:0>3}".format(str(count)) +".jpg"  # 创建新的文件名称
                    new_path = os.path.join(path2, new_name)
                    os.rename(folder_path, new_path)  # 重命名文件
                    count += 1
                    print(new_path)