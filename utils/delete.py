import os
import shutil

def remove_files_and_dirs(base_dir, dirs_to_remove, files_to_remove):
    # 删除指定的目录
    for dir_name in dirs_to_remove:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"目录 {dir_name} 已删除。")
        else:
            print(f"目录 {dir_name} 不存在。")

    # 删除指定的文件
    for file_name in files_to_remove:
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"文件 {file_name} 已删除。")
        else:
            print(f"文件 {file_name} 不存在。")

if __name__ == "__main__":
    # obj_ids = [1,5,6,8,9,10,11,12]
    base_dir = '/home/mendax/project/Datasets/lm/000012'  # 设置基础目录路径
    base_dir = '/home/mendax/project/Datasets/lmo/test/000002'  # 设置基础目录路径
    dirs_to_remove = ['mincrop', 'centercrop', 'squarecrop', 'depthcrop', 'newview', 'newviewcrop']  # 需要删除的目录
    files_to_remove = ['object_info.json']  # 需要删除的文件

    remove_files_and_dirs(base_dir, dirs_to_remove, files_to_remove)
