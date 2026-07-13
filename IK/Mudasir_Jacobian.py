import numpy as np

def forward_kinematics(theta1, theta2, theta3, l1, l2, l3):
   
    a1 = theta1
    a2 = theta1 + theta2
    a3 = theta1 + theta2 + theta3
    
    x = l1 * np.cos(a1) + l2 * np.cos(a2) + l3 * np.cos(a3)
    y = l1 * np.sin(a1) + l2 * np.sin(a2) + l3 * np.sin(a3)
    phi = theta1 + theta2 + theta3
    return np.array([x, y, phi])

def ik_jacobian(x_tgt, y_tgt, phi_tgt, l1, l2, l3, theta_init, max_iter=100, tol=1e-5):
   
    X_target = np.array([x_tgt, y_tgt, phi_tgt])
    
   
    theta = np.array(theta_init, dtype=float)
    
    for i in range(max_iter):
       
        X_current = forward_kinematics(theta[0], theta[1], theta[2], l1, l2, l3)
        
       
        error = X_target - X_current
        
        
        if np.linalg.norm(error) < tol:
            print(f"[Jacobian converged in {i} iterations]")
            return theta
            
        
        s1 = np.sin(theta[0])
        s12 = np.sin(theta[0] + theta[1])
        s123 = np.sin(theta[0] + theta[1] + theta[2])
        
        c1 = np.cos(theta[0])
        c12 = np.cos(theta[0] + theta[1])
        c123 = np.cos(theta[0] + theta[1] + theta[2])
        
        # 
        J = np.array([
            [-l1*s1 - l2*s12 - l3*s123, -l2*s12 - l3*s123, -l3*s123],
            [ l1*c1 + l2*c12 + l3*c123,  l2*c12 + l3*c123,  l3*c123],
            [ 1.0,                       1.0,               1.0]
        ])
        
       
        J_pinv = np.linalg.pinv(J)
        
        
        theta += J_pinv @ error

        theta = np.arctan2(np.sin(theta), np.cos(theta))
        
    raise RuntimeError("Jacobian method FAILED to converge within maximum iterations.")



L1, L2, L3 = 15.0, 12.0, 8.0
    

target_x = 20.0
target_y = 15.0
target_phi = np.radians(45.0)  # 45 degrees orientation

print(f"--- RUNNING CODES FOR TARGET: X={target_x}, Y={target_y}, Phi=45° ---\n")





initial_guess = [np.radians(10), np.radians(10), np.radians(10)]
angles_jac = ik_jacobian(target_x, target_y, target_phi, L1, L2, L3, initial_guess)
print("2. Jacobian (Numerical Iterative) Solutions (in Degrees):")
print(f"   Theta 1: {np.degrees(angles_jac[0]):.2f}°")
print(f"   Theta 2: {np.degrees(angles_jac[1]):.2f}°")
print(f"   Theta 3: {np.degrees(angles_jac[2]):.2f}°\n")


