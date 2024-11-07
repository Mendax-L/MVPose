from rot_train import train_rot
from depth_train import train_depth
from centeruv_train import train_centeruv


if __name__ == "__main__":
    # 训练yolov8seg    
    # print("Starting training for yolov8seg ...")
    # run_script("seg_net.py")
    obj_ids = [12]
    # 训练 rot_net
    # print("Starting training for rot_net...")
    # train_rot(obj_ids)
    
    # # 训练 depth_net
    # print("Starting training for depth_net...")
    # train_depth(obj_ids)
    
    # 训练 centerxy_net
    print("Starting training for centeruv_net...")
    train_centeruv(obj_ids)
    
    print("All models have been trained successfully.")
