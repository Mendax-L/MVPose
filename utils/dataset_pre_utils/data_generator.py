import blenderproc as bproc
import argparse
import numpy as np
import sys
import os
import shutil




def random_rotation_matrix():
    # 生成随机的欧拉角
    theta_x = np.random.uniform(0, 2 * np.pi)
    theta_y = np.random.uniform(0, 2 * np.pi)
    theta_z = np.random.uniform(0, 2 * np.pi)

    # 计算每个轴的旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    
    # 组合旋转矩阵
    R = Rz @ Ry @ Rx
    return R




def random_transform_matrix(z_range=(0.3, 1.5)):
    rotation_matrix = random_rotation_matrix()
    z_translation = np.random.uniform(z_range[0], z_range[1])
    
    # 构造4x4变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[2, 3] = z_translation  # z方向的平移
    return transform_matrix





sys.path.append('/home/mendax/anaconda3/envs/pose/lib/python3.11/site-packages')

parser = argparse.ArgumentParser()
parser.add_argument('object', nargs='?', default="/home/mendax/project/SATPose/datasets/lmo/models/obj_000001.ply", help="Path to the model file")
parser.add_argument('output_dir', nargs='?', default="/home/mendax/project/SATPose/datasets/lmo/pbr", help="Path to where the final files will be saved")
args = parser.parse_args()


# 删除旧的输出目录（如果存在）
if os.path.exists(args.output_dir):
    shutil.rmtree(args.output_dir)

# 重新创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

bproc.init()

# load the objects into the scene
obj = bproc.loader.load_obj(args.object)[0]
# Use vertex color for texturing
for mat in obj.get_materials():
    mat.map_vertex_color()
# Set pose of object via local-to-world transformation matrix


# Set category id which will be used in the BopWriter
obj.set_cp("category_id", 1)

# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([5, -5, 5])
light.set_energy(1000)

# Set intrinsics via K matrix
bproc.camera.set_intrinsics_from_K_matrix(
    [[572.4114, 0.0, 325.2611],
     [0.0, 573.57043, 242.04899],
     [0.0, 0.0, 1.0]], 640, 480
)
# Set camera pose via cam-to-world transformation matrix
cam2world = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
# Change coordinate frame of transformation matrix from OpenCV to Blender coordinates
cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world, ["X", "-Y", "-Z"])
for i in range(5):
    transform_matrix = random_transform_matrix()
    print(transform_matrix)
    obj.set_local2world_mat(transform_matrix)    
    # Scale 3D model from mm to m
    obj.set_scale([0.001, 0.001, 0.001])
    bproc.camera.add_camera_pose(cam2world)

# activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# render the whole pipeline
data = bproc.renderer.render()

# Write object poses, color and depth in bop format
bproc.writer.write_bop(args.output_dir, [obj], data["depth"], data["colors"], m2mm=True, append_to_existing_output=True)