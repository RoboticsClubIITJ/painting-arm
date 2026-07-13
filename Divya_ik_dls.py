import numpy as np

def forward_kinematics(q, l1, l2, l3):
    """Compute end-effector pose [x, y, phi] for 3-link planar arm."""
    t1, t2, t3 = q
    x = l1*np.cos(t1) + l2*np.cos(t1+t2) + l3*np.cos(t1+t2+t3)
    y = l1*np.sin(t1) + l2*np.sin(t1+t2) + l3*np.sin(t1+t2+t3)
    phi = t1 + t2 + t3
    return np.array([x, y, phi])

def jacobian(q, l1, l2, l3):
    """Compute 3x3 Jacobian for 3-link planar arm."""
    t1, t2, t3 = q
    J = np.zeros((3,3))

    # ∂x/∂θ1, ∂x/∂θ2, ∂x/∂θ3
    J[0,0] = -l1*np.sin(t1) - l2*np.sin(t1+t2) - l3*np.sin(t1+t2+t3)
    J[0,1] = -l2*np.sin(t1+t2) - l3*np.sin(t1+t2+t3)
    J[0,2] = -l3*np.sin(t1+t2+t3)

    # ∂y/∂θ1, ∂y/∂θ2, ∂y/∂θ3
    J[1,0] =  l1*np.cos(t1) + l2*np.cos(t1+t2) + l3*np.cos(t1+t2+t3)
    J[1,1] =  l2*np.cos(t1+t2) + l3*np.cos(t1+t2+t3)
    J[1,2] =  l3*np.cos(t1+t2+t3)

    # ∂φ/∂θ1, ∂φ/∂θ2, ∂φ/∂θ3
    J[2,0] = 1
    J[2,1] = 1
    J[2,2] = 1

    return J

def ik_damped_least_squares(xd, yd, phid, l1, l2, l3, q0, lam=0.1, tol=1e-3, max_iter=1000):
    """Inverse kinematics using Damped Least Squares method."""
    q = np.array(q0, dtype=float)
    target = np.array([xd, yd, phid])

    for i in range(max_iter):
        pose = forward_kinematics(q, l1, l2, l3)
        e = target - pose  # error vector
        if np.linalg.norm(e) < tol:
            break
        J = jacobian(q, l1, l2, l3)

        # Damped least squares update
        JJt = J @ J.T
        lamI = (lam**2) * np.eye(JJt.shape[0])
        dq = J.T @ np.linalg.inv(JJt + lamI) @ e

        q += dq

    return q, pose

# --- Example run ---
if __name__ == "__main__":
    # Link lengths
    l1, l2, l3 = 1.0, 1.0, 1.0

    # Desired pose (x, y, phi)
    xd, yd, phid = 1.5, 1.0, np.radians(90)

    # Initial guess for joint angles
    q0 = [0.0, 0.0, 0.0]

    q_sol, pose_sol = ik_damped_least_squares(xd, yd, phid, l1, l2, l3, q0)

    desired_pose = np.array([xd, yd, phid])
    error = pose_sol - desired_pose

    print("Solved joint angles (rad):", q_sol)
    print("End-effector pose reached:", pose_sol)
    print("Desired pose:", desired_pose)
    print("Error vector (reached - desired):", error)
    print("Error norm:", np.linalg.norm(error))
