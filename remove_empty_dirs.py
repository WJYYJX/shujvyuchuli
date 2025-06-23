import os

def remove_empty_dirs(directory):
    # 遍历目录下的所有文件和子目录
    for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
        # 如果文件夹为空，则删除它
        if not os.listdir(dirpath):
            print(f"Removing empty directory: {dirpath}")
            os.rmdir(dirpath)

# 指定你要清理的目录路径
directory = 'C:\\Users\\111\\Desktop\\new1'

# 调用函数删除空文件夹
remove_empty_dirs(directory)