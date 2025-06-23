import os
import shutil


def main():
    # 直接在代码中设置目标路径（使用原始字符串）
    target_dir = r"F:\goubangzi_1.2.4.6.10.16"

    # Windows长路径处理
    if not target_dir.startswith("\\\\?\\"):
        target_dir = "\\\\?\\" + os.path.abspath(target_dir)

    # 路径有效性验证
    if not os.path.exists(target_dir):
        print(f"路径不存在: {target_dir}")
        return

    if not os.path.isdir(target_dir):
        print(f"目标不是目录: {target_dir}")
        return

    # 执行主逻辑
    process_directory(target_dir)


def process_directory(root_dir):
    """处理三级目录的核心逻辑"""
    try:
        for root, dirs, files in os.walk(root_dir):
            if is_third_level(root_dir, root):
                print(f"\n正在处理三级目录：{root}")
                move_files(root)
                clean_empty_dirs(root)
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")


def is_third_level(base_path, current_path):
    """判断是否为三级子目录"""
    relative = os.path.relpath(current_path, base_path)
    return relative.count(os.sep) == 2


def move_files(third_dir):
    """移动文件到三级目录"""
    for item in os.listdir(third_dir):
        item_path = os.path.join(third_dir, item)

        if os.path.isdir(item_path):
            for root, _, files in os.walk(item_path):
                for file in files:
                    src = win_long_path(os.path.join(root, file))
                    dest = unique_dest_path(third_dir, file)

                    try:
                        shutil.move(src, dest)
                        print(f"移动成功: {src} → {dest}")
                    except PermissionError:
                        print(f"权限不足跳过: {src}")
                    except shutil.Error as e:
                        print(f"移动错误: {str(e)}")


def unique_dest_path(directory, filename):
    """生成唯一文件名"""
    base, ext = os.path.splitext(filename)
    counter = 1
    dest = win_long_path(os.path.join(directory, filename))

    while os.path.exists(dest):
        new_name = f"{base}_{counter}{ext}"
        dest = win_long_path(os.path.join(directory, new_name))
        counter += 1
    return dest


def win_long_path(path):
    """处理Windows长路径"""
    return "\\\\?\\" + os.path.abspath(path) if len(path) > 260 else path


def clean_empty_dirs(start_dir):
    """清理空目录"""
    for root, dirs, files in os.walk(start_dir, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                if not os.listdir(win_long_path(dir_path)):
                    os.rmdir(win_long_path(dir_path))
                    print(f"已删除空目录: {dir_path}")
            except OSError as e:
                print(f"删除失败: {dir_path} - {str(e)}")


if __name__ == "__main__":
    main()
    print("\n操作完成，请检查以下内容：")
    print("1. 所有文件移动结果")
    print("2. 剩余目录结构")
    input("按回车键退出...")