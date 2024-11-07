import torch

# Psi function (Ψ) to compute the rotation matrix that aligns vector p with vector q
def psi(p, q):
    r = torch.cross(p, q)  # Compute the cross product of p and q
    r_norm_sq = torch.dot(r, r)  # Compute the norm squared of r
    p_dot_q = torch.dot(p, q)  # Compute the dot product of p and q
    
    # Compute the rotation matrix
    I = torch.eye(3)  # 3x3 identity matrix
    r_cross = torch.tensor([
        [0, -r[2], r[1]],
        [r[2], 0, -r[0]],
        [-r[1], r[0], 0]
    ])  # Skew-symmetric matrix of r
    
    rotation_matrix = I + r_cross + (r_cross @ r_cross) / (1 + p_dot_q)
    
    return rotation_matrix

# Given camera intrinsic matrix Kc and the object center projection c in the image
def recover_egocentric_pose(Kc, c, viewpoint_v, d):
    # Step 1: Compute the ray from the camera through the object center projection
    p = torch.tensor([0, 0, 1], dtype=torch.float32)  # Camera principal axis
    c_homogeneous = torch.cat((c, torch.tensor([1.0])))  # Convert c to homogeneous coordinates
    q = torch.inverse(Kc) @ c_homogeneous  # Kc^{-1} * c

    # Step 2: Compute Rc, the rotation between the camera principal axis and the ray
    Rc = psi(p, q)

    # Step 3: Compute Rv, the rotation matrix for the viewpoint v (for simplicity, assume identity)
    Rv = viewpoint_v  # Rotation matrix for viewpoint (assumed to be a tensor)
    
    # Step 4: Compute the object pose PE in SE(3)
    R = Rc @ Rv  # Combined rotation
    t = Rc @ torch.tensor([0, 0, d], dtype=torch.float32)  # Translation vector

    # Step 5: Construct the 4x4 pose matrix PE
    PE = torch.eye(4)
    PE[:3, :3] = R
    PE[:3, 3] = t

    return PE

if __name__=="__main__":
    # Example usage
    Kc = torch.tensor([[1000.0, 0, 320.0], [0, 1000.0, 240.0], [0, 0, 1.0]])  # Example camera intrinsic matrix
    c = torch.tensor([320.0, 240.0])  # Example object center projection in image
    viewpoint_v = torch.eye(3)  # Example viewpoint rotation matrix (identity in this case)
    d = 5.0  # Example distance from camera

    pose_matrix = recover_egocentric_pose(Kc, c, viewpoint_v, d)
    print("Recovered Pose Matrix PE:\n", pose_matrix)
