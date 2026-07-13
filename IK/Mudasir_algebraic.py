import numpy as np

def ik_algebraic(x_tgt, y_tgt, phi_tgt, l1, l2, l3, posture="elbow_up"):
    
   
    x2 = x_tgt - l3 * np.cos(phi_tgt)
    y2 = y_tgt - l3 * np.sin(phi_tgt)
    
  
    r_sq = x2**2 + y2**2
    cos_theta2 = (r_sq - l1**2 - l2**2) / (2 * l1 * l2)
    
    if cos_theta2 < -1.0 or cos_theta2 > 1.0:
        raise ValueError("Its out of reach")
        
 
        theta2 = np.arccos(cos_theta2)     
    else:
        theta2 = -np.arccos(cos_theta2) 
        
    
    theta1 = np.arctan2(y2, x2) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    
   
    theta3 = phi_tgt - theta1 - theta2
    
    return np.array([theta1, theta2, theta3])
