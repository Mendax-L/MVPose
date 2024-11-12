import os
import shutil

def remove_files_and_dirs(base_dir, dirs_to_remove, files_to_remove):
    # 删除指定的目录
    for dir_name in dirs_to_remove:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"目录 {dir_path} 已删除。")
        else:
            print(f"目录 {dir_path} 不存在。")

    # 删除指定的文件
    for file_name in files_to_remove:
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"文件 {file_path} 已删除。")
        else:
            print(f"文件 {file_path} 不存在。")

if __name__ == "__main__":
    obj_ids = [1,5,6,8,9,10,11,12]
    obj_ids = [1]
    for scene_id in range(0, 92):
        target_dir = f'../Datasets/ycbv/train_real/{str(scene_id).zfill(6)}'
        if os.path.exists(target_dir):
        # base_dir = f'/home/mendax/project/Datasets/lm/{str(obj_id).zfill(6)}'  # 设置基础目录路径
            base_dir = target_dir  # 设置基础目录路径
            dirs_to_remove = ['view_000/crop']  # 需要删除的目录
            files_to_remove = ['view_000.json']  # 需要删除的文件

            remove_files_and_dirs(base_dir, dirs_to_remove, files_to_remove)
