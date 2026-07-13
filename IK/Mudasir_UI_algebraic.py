import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider



def ik_algebraic(x_tgt, y_tgt, phi_tgt, l1, l2, l3):
    """Solves IK instantly using geometric decoupling and Law of Cosines."""
   
    x2 = x_tgt - l3 * np.cos(phi_tgt)
    y2 = y_tgt - l3 * np.sin(phi_tgt)
    
  
    r_sq = x2**2 + y2**2
    cos_theta2 = (r_sq - l1**2 - l2**2) / (2 * l1 * l2)
    
    if cos_theta2 < -1.0 or cos_theta2 > 1.0:
        raise ValueError("Target out of reach")
        

    theta2 = np.arccos(cos_theta2)
    theta1 = np.arctan2(y2, x2) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    theta3 = phi_tgt - theta1 - theta2
    
    return theta1, theta2, theta3

def forward_kinematics_visual(theta1, theta2, theta3, l1, l2, l3):
    """Calculates all individual joint coordinates for line plotting."""
    angle1 = theta1
    angle2 = theta1 + theta2
    angle3 = theta1 + theta2 + theta3
    
    x0, y0 = 0.0, 0.0
    x1 = l1 * np.cos(angle1)
    y1 = l1 * np.sin(angle1)
    x2 = x1 + l2 * np.cos(angle2)
    y2 = y1 + l2 * np.sin(angle2)
    x3 = x2 + l3 * np.cos(angle3)
    y3 = y2 + l3 * np.sin(angle3)
    
    return [x0, x1, x2, x3], [y0, y1, y2, y3]



L1, L2, L3 = 15.0, 12.0, 8.0
max_reach = L1 + L2 + L3


fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(bottom=0.25)


init_x, init_y, init_phi = 18.0, 12.0, np.radians(0.0)
t1, t2, t3 = ik_algebraic(init_x, init_y, init_phi, L1, L2, L3)
x_j, y_j = forward_kinematics_visual(t1, t2, t3, L1, L2, L3)


arm_line, = ax.plot(x_j, y_j, 'o-', linewidth=5, markersize=10, color='blue', label='Robotic Arm')
target_dot, = ax.plot(init_x, init_y, 'rx', markersize=12, markeredgewidth=3, label='Target Pixel')
ax.plot(0, 0, 'ks', markersize=12) 


ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(-max_reach - 5, max_reach + 5)
ax.set_ylim(-max_reach - 5, max_reach + 5)
ax.set_aspect('equal')
ax.legend(loc='upper right')


angle_text = ax.text(-max_reach, max_reach + 2, "", fontsize=10, color="purple", weight="bold")


ax_x = plt.axes([0.15, 0.15, 0.7, 0.03])
ax_y = plt.axes([0.15, 0.10, 0.7, 0.03])
ax_p = plt.axes([0.15, 0.05, 0.7, 0.03])

slider_x = Slider(ax_x, 'Target X', -max_reach, max_reach, valinit=init_x, valfmt='%.1f')
slider_y = Slider(ax_y, 'Target Y', -max_reach, max_reach, valinit=init_y, valfmt='%.1f')
slider_p = Slider(ax_p, 'Target Phi°', -180.0, 180.0, valinit=0.0, valfmt='%.1f')




def update(val):

    x_tgt = slider_x.val
    y_tgt = slider_y.val
    phi_tgt = np.radians(slider_p.val) 
    
   
    target_dot.set_data([x_tgt], [y_tgt])
    
    try:
      
        theta1, theta2, theta3 = ik_algebraic(x_tgt, y_tgt, phi_tgt, L1, L2, L3)
        
       
        x_coords, y_coords = forward_kinematics_visual(theta1, theta2, theta3, L1, L2, L3)
        
      
        arm_line.set_data(x_coords, y_coords)
        arm_line.set_color('blue')
        
        
        angle_text.set_text(
            f"IK Output Angles:\n"
            f"Joint 1: {np.degrees(theta1):.1f}°\n"
            f"Joint 2: {np.degrees(theta2):.1f}°\n"
            f"Joint 3: {np.degrees(theta3):.1f}°"
        )
        angle_text.set_color("purple")
        ax.set_title("3-Link Algebraic IK Interactive Simulator", color="black", weight="bold")
        
    except ValueError:
       
        arm_line.set_color('red') 
        angle_text.set_text("IK Error:\nTARGET OUT OF REACH")
        angle_text.set_color("red")
        ax.set_title("⚠️ OUT OF WORKSPACE REGION ⚠️", color="red", weight="bold")
        
    
    fig.canvas.draw_idle()


slider_x.on_changed(update)
slider_y.on_changed(update)
slider_p.on_changed(update)


update(None)
plt.show()
