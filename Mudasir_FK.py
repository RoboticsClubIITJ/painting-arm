import numpy
import matplotlib.pyplot as plt

def forward_kinematics_visual(theta1, theta2, theta3, l1, l2, l3):
   
    #calculaton etc
    q1 = numpy.radians(theta1)
    q2 = numpy.radians(theta2)
    q3 = numpy.radians(theta3)
    

    angle1 = q1
    angle2 = q1 + q2
    angle3 = q1 + q2 + q3
    
 
    x0, y0 = 0.0, 0.0
    

    x1 = l1 * numpy.cos(angle1)
    y1 = l1 * numpy.sin(angle1)
    
   
    x2 = x1 + l2 * numpy.cos(angle2)
    y2 = y1 + l2 * numpy.sin(angle2)
    
    
    x3 = x2 + l3 * numpy.cos(angle3)
    y3 = y2 + l3 * numpy.sin(angle3)
    
    
    x_coords = [x0, x1, x2, x3]
    y_coords = [y0, y1, y2, y3]
    
    return x_coords, y_coords
#took help of AI for GUI
def plot_arm(x_coords, y_coords, max_reach):
    
    plt.figure(figsize=(6, 6))
    
    
    plt.plot(x_coords, y_coords, 'o-', linewidth=4, markersize=10, color='blue', label='Robotic Arm')
    
    
    plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=12, label='Paintbrush Tip')
    
   
    plt.plot(0, 0, 'ks', markersize=12, label='Base Anchor')
    
   
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("Week 1: 3-Link Arm Forward Kinematics Visualizer", fontsize=12, fontweight='bold')
    plt.xlabel("X Canvas Coordinate", fontsize=10)
    plt.ylabel("Y Canvas Coordinate", fontsize=10)
    
    
    plt.xlim(-max_reach - 5, max_reach + 5)
    plt.ylim(-max_reach - 5, max_reach + 5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right')
    
  
    plt.show()



  
L1, L2, L3 = 15.0, 12.0, 8.0
max_reach = L1 + L2 + L3 


target_theta1 = 45  
target_theta2 = -30  
target_theta3 = -15  


x_joints, y_joints = forward_kinematics_visual(target_theta1, target_theta2, target_theta3, L1, L2, L3)


print(f"Base: ({x_joints[0]}, {y_joints[0]})")
print(f"Joint 1: ({x_joints[1]:.2f}, {y_joints[1]:.2f})")
print(f"Joint 2: ({x_joints[2]:.2f}, {y_joints[2]:.2f})")
print(f"Paintbrush: ({x_joints[3]:.2f}, {y_joints[3]:.2f})")


plot_arm(x_joints, y_joints, max_reach)
