import bpy
import random
import math
import mathutils
import os
import config

def load_blend_file(filepath):
    bpy.ops.wm.open_mainfile(filepath=filepath)

def setup_camera():
    camera = bpy.data.objects.get("Camera")
    if not camera:
        camera_data = bpy.data.cameras.new(name="Camera")
        camera = bpy.data.objects.new("Camera", camera_data)
        bpy.context.collection.objects.link(camera)
    
    camera.location = config.camera_location
    camera.rotation_euler = config.camera_rotation_euler
    camera.data.lens = config.focal_length
    camera.data.sensor_width = config.sensor_width
    camera.data.sensor_height = config.sensor_height
    camera.data.shift_x = (config.principal_point[0] - 0.5) * 2
    camera.data.shift_y = (config.principal_point[1] - 0.5) * 2
    return camera

def setup_scene(camera):
    scene = bpy.context.scene
    render = scene.render
    render.engine = config.render_engine
    render.resolution_x = config.resolution_x
    render.resolution_y = config.resolution_y
    render.resolution_percentage = config.resolution_percentage
    return scene

def position_objects(lego, background, camera):
    background.location = (0, 0, 0)
    lego_z = random.uniform(0, config.lego_z_most)
    scale = (camera.location.z - lego_z) / (camera.location.z - config.lego_z_most)
    lego.location = (random.uniform(-0.09 * scale, -0.1 * scale), random.uniform(-0.06 * scale, -0.04 * scale), lego_z)
    lego.rotation_mode = 'XYZ'
    lego.rotation_euler = (random.uniform(-math.pi / 3, math.pi / 3), random.uniform(-math.pi / 3, math.pi / 3), random.uniform(0, 2 * math.pi))

def calculate_bbox(lego, bounding_box, camera, resolution_x, resolution_y):
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')
    
    for vertex in lego.bound_box:
        min_x = min(min_x, vertex[0])
        max_x = max(max_x, vertex[0])
        min_y = min(min_y, vertex[1])
        max_y = max(max_y, vertex[1])
    
    bbox_size = max((max_x - min_x), (max_y - min_y)) / 1.5
    bounding_box.scale = (bbox_size, bbox_size, 0)
    bounding_box.location = lego.location

    camera_matrix = camera.matrix_world
    bbox_size_coords = mathutils.Vector((bbox_size, 0, bounding_box.location.z))
    world_coords = mathutils.Vector((bounding_box.location))

    K = mathutils.Matrix(config.K)

    uv = K @ camera_matrix.inverted() @ world_coords / (bounding_box.location.z - camera.location.z)
    bbox_uv = K @ camera_matrix.inverted() @ bbox_size_coords / (bounding_box.location.z - camera.location.z)

    bbox_size_incamera = bbox_uv.x - resolution_x / 2

    x1, y1 = max(uv[0] - bbox_size_incamera, 0), max(uv[1] - bbox_size_incamera, 0)
    x2, y2 = min(uv[0] + bbox_size_incamera, resolution_x), min(uv[1] + bbox_size_incamera, resolution_y)
    
    bounding_box.display_type = 'WIRE'
    bounding_box.show_in_front = True

    z_c = bounding_box.location.z - camera.location.z 
    x_c = abs(z_c) * (x1 -  resolution_x / 2) * config.sensor_width / (config.focal_length * resolution_x)
    y_c = abs(z_c) * (-y1 +  resolution_y / 2) * config.sensor_height / (config.focal_length * resolution_y)

    world_coords = camera_matrix @ mathutils.Vector((x_c, y_c, z_c))
    bpy.ops.mesh.primitive_cube_add(size=0.005, location=world_coords)

def get_6d_pose(camera, lego):
    camera_matrix_world = camera.matrix_world
    lego_matrix_world = lego.matrix_world
    camera_matrix_world_inv = camera_matrix_world.inverted()
    relative_matrix = camera_matrix_world_inv @ lego_matrix_world
    relative_location = relative_matrix.to_translation()
    relative_rotation = relative_matrix.to_3x3()
    return relative_location, relative_rotation

def main():
    os.chdir(config.current_directory)
    load_blend_file(config.blend_file_path)

    lego = bpy.data.objects.get("VGA_Port_Female_PL")
    background = bpy.data.objects.get("background")
    bounding_box = bpy.data.objects.get("BBox")
    if not lego or not background:
        raise Exception("Target or background not found!")

    camera = setup_camera()
    scene = setup_scene(camera)
    position_objects(lego, background, camera)

    calculate_bbox(lego, bounding_box, camera, config.resolution_x, config.resolution_y)
    relative_location, relative_rotation = get_6d_pose(camera, lego)
    print(relative_location)
    print(relative_rotation)

if __name__ == "__main__":
    main()
