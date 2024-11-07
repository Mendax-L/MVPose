import numpy as np

# Function Ψ(p, q) to compute the rotation matrix that aligns vector p with vector q
def psi(p, q):
    r = np.cross(p, q)  # Compute the cross product of p and q
    r_norm_sq = np.dot(r, r)  # Compute the norm squared of r
    p_dot_q = np.dot(p, q)  # Compute the dot product of p and q
    
    # Compute the rotation matrix
    I = np.eye(3)  # 3x3 identity matrix
    r_cross = np.array([
        [0, -r[2], r[1]],
        [r[2], 0, -r[0]],
        [-r[1], r[0], 0]
    ])  # Skew-symmetric matrix of r
    
    rotation_matrix = I + r_cross + (r_cross @ r_cross) / (1 + p_dot_q)
    
    return rotation_matrix

# Given camera intrinsic matrix Kc and the object center projection c in the image
def recover_egocentric_pose(Kc, c, viewpoint_v, d):
    # Step 1: Compute the ray from the camera through the object center projection
    p = np.array([0, 0, 1])  # Camera principal axis
    q = np.linalg.inv(Kc) @ np.append(c, 1)  # Kc^{-1} * c

    # Step 2: Compute Rc, the rotation between the camera principal axis and the ray
    Rc = psi(p, q)

    # Step 3: Compute Rv, the rotation matrix for the viewpoint v (for simplicity, assume identity)
    Rv = np.eye(3)  # Placeholder, can be replaced with actual viewpoint rotation matrix
    
    # Step 4: Compute the object pose PE in SE(3)
    R = Rc @ Rv  # Combined rotation
    t = Rc @ np.array([0, 0, d])  # Translation vector

    # Step 5: Construct the 4x4 pose matrix PE
    PE = np.eye(4)
    PE[:3, :3] = R
    PE[:3, 3] = t

    return PE

if __name__=="__main__":
    # Example usage
    Kc = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])  # Example camera intrinsic matrix
    c = np.array([320, 240])  # Example object center projection in image
    viewpoint_v = np.eye(3)  # Example viewpoint rotation matrix (identity in this case)
    d = 5.0  # Example distance from camera

    pose_matrix = recover_egocentric_pose(Kc, c, viewpoint_v, d)
    print("Recovered Pose Matrix PE:\n", pose_matrix)
