import tkinter as tk
from tkinter import messagebox
import numpy as np

# Embedding Matplotlib into Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



# ==========================================
# 1. YOUR CORE MATH FUNCTION
# ==========================================
def calculateIK(l1,l2,l3,targetx,targety,finalangle):

    afin=np.radians(finalangle)

    # find wrist coordinates

    xw=targetx-l3*np.cos(afin)
    yw=targety-l3*np.sin(afin)

    # xw=l1*c1+l2*c12
    # yw=l1*s1+l2*s12
    # by squaring both and adding 

    c2 = (xw**2 + yw**2 - l1**2 - l2**2)/(2*l1*l2)

    if abs(c2)>1:
        print("c2 out of bound")
        return None,None,None
    
    theta2 = np.arccos(c2)

    # theta1 will be derived from traingle formed between all three joints
    # i.e, thetawrist - inner triangle angle
    
    theta1 = np.arctan2(yw,xw) - np.arctan2(l2*np.sin(theta2),l1+l2*np.cos(theta2))

    theta3 = afin - theta1- theta2

    # Added return statement so the GUI application can access the results
    return theta1, theta2, theta3

# ==========================================
# 2. INTERACTION & SIMULATION LOGIC
# ==========================================
def on_calculate_and_plot():
    try:
        # Get numeric values from GUI inputs
        l1 = float(entry_l1.get())
        l2 = float(entry_l2.get())
        l3 = float(entry_l3.get())
        
        targetx = float(entry_tx.get())
        targety = float(entry_ty.get())
        finalangle = float(entry_fa.get())
        
        # 1. Run your exact Inverse Kinematics function
        t1, t2, t3 = calculateIK(l1, l2, l3, targetx, targety, finalangle)
        
        # If out of bounds, trigger error pop-up window
        if t1 is None:
            messagebox.showerror("Kinematics Error", "Target position is out of the robot's physical reach boundary!")
            return
            
        # Convert the calculated radians back to degrees for human-readable labels
        deg1 = np.degrees(t1)
        deg2 = np.degrees(t2)
        deg3 = np.degrees(t3)
        
        # Update the text labels on the GUI panel
        label_result_a1.config(text=f"Angle 1 (Shoulder): {deg1:.2f}°")
        label_result_a2.config(text=f"Angle 2 (Elbow): {deg2:.2f}°")
        label_result_a3.config(text=f"Angle 3 (Wrist): {deg3:.2f}°")
        
        # 2. Reconstruct joint coordinates chronologically for visualization 
        x0, y0 = 0.0, 0.0
        x1 = l1 * np.cos(t1)
        y1 = l1 * np.sin(t1)
        x2 = x1 + l2 * np.cos(t1 + t2)
        y2 = y1 + l2 * np.sin(t1 + t2)
        x3, y3 = targetx, targety  # End effector lands exactly at target point
        
        # 3. Update the Matplotlib Plot Live
        ax.clear()  
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        
        # Plot physical linkages mesh wireframe
        ax.plot([x0, x1, x2, x3], [y0, y1, y2, y3], color='#2c3e50', linewidth=4, marker='o', 
                markersize=8, markerfacecolor='#e74c3c', markeredgecolor='black', label="Arm Mesh Links")
        
        # Highlight target coordinate location dot
        ax.plot(x3, y3, marker='X', markersize=10, markerfacecolor='#e67e22', markeredgecolor='black', label="Target Point")
        
        # Dynamically fit grid sizing envelope constraints based on lengths
        max_reach = l1 + l2 + l3 + 5
        ax.set_xlim(-max_reach, max_reach)
        ax.set_ylim(-max_reach, max_reach)
        ax.set_aspect('equal', adjustable='box')  
        ax.set_title("3-DOF Inverse Kinematics Real-time Simulation Mesh", fontsize=10, fontweight='bold')
        ax.legend(loc="upper right")
        
        # Draw canvas update refresh frame
        canvas.draw()
        
    except ValueError:
        messagebox.showerror("Input Error", "Please fill all input fields with valid numbers!")

def reset_lengths():
    entry_l1.delete(0, tk.END)
    entry_l2.delete(0, tk.END)
    entry_l3.delete(0, tk.END)

def reset_targets():
    entry_tx.delete(0, tk.END)
    entry_ty.delete(0, tk.END)
    entry_fa.delete(0, tk.END)
    
    # Re-initialize output display configurations
    label_result_a1.config(text="Angle 1 (Shoulder): --")
    label_result_a2.config(text="Angle 2 (Elbow): --")
    label_result_a3.config(text="Angle 3 (Wrist): --")
    ax.clear()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_aspect('equal', adjustable='box')
    canvas.draw()

# ==========================================
# 3. WINDOW & GRID BUILDING BLOCK LAYOUT
# ==========================================
root = tk.Tk()
root.title("Inter-IIT: Bob Ross 3-DOF Robot Arm Inverse Kinematics Solver")
root.geometry("950x520")

# Setup container panels side-by-side
left_control_panel = tk.Frame(root, padx=15, pady=10)
left_control_panel.pack(side="left", fill="y")

right_simulation_panel = tk.Frame(root, bg="white", padx=5, pady=5)
right_simulation_panel.pack(side="right", fill="both", expand=True)

# --- LEFT PANEL CONFIGURATION ---

# Frame 1: Link Structural Scaling Parameters
frame_lengths = tk.LabelFrame(left_control_panel, text=" 1. Link Dimensions (cm) ", padx=10, pady=10, font=("Arial", 10, "bold"))
frame_lengths.pack(fill="x", pady=5)

tk.Label(frame_lengths, text="Link 1:").grid(row=0, column=0, sticky="w", pady=2)
entry_l1 = tk.Entry(frame_lengths, width=8)
entry_l1.grid(row=0, column=1, padx=5, pady=2)

tk.Label(frame_lengths, text="Link 2:").grid(row=1, column=0, sticky="w", pady=2)
entry_l2 = tk.Entry(frame_lengths, width=8)
entry_l2.grid(row=1, column=1, padx=5, pady=2)

tk.Label(frame_lengths, text="Link 3:").grid(row=2, column=0, sticky="w", pady=2)
entry_l3 = tk.Entry(frame_lengths, width=8)
entry_l3.grid(row=2, column=1, padx=5, pady=2)

btn_reset_l = tk.Button(frame_lengths, text="Reset Lengths", command=reset_lengths, fg="white", bg="#c0392b")
btn_reset_l.grid(row=1, column=2, padx=15)

# Frame 2: Operational Target Profiles Setup
frame_targets = tk.LabelFrame(left_control_panel, text=" 2. Coordinate Parameters target ", padx=10, pady=10, font=("Arial", 10, "bold"))
frame_targets.pack(fill="x", pady=5)

tk.Label(frame_targets, text="Target X:").grid(row=0, column=0, sticky="w", pady=2)
entry_tx = tk.Entry(frame_targets, width=8)
entry_tx.grid(row=0, column=1, padx=5, pady=2)

tk.Label(frame_targets, text="Target Y:").grid(row=1, column=0, sticky="w", pady=2)
entry_ty = tk.Entry(frame_targets, width=8)
entry_ty.grid(row=1, column=1, padx=5, pady=2)

tk.Label(frame_targets, text="Final Angle (°):").grid(row=2, column=0, sticky="w", pady=2)
entry_fa = tk.Entry(frame_targets, width=8)
entry_fa.grid(row=2, column=1, padx=5, pady=2)

btn_reset_t = tk.Button(frame_targets, text="Reset Profile", command=reset_targets, fg="white", bg="#c0392b")
btn_reset_t.grid(row=1, column=2, padx=15)

# Computation Engine Execution Button Trigger
btn_plot = tk.Button(left_control_panel, text="Compute & Render Inverse Kinematics", command=on_calculate_and_plot, 
                     bg="#e67e22", fg="white", font=("Arial", 11, "bold"), pady=6)
btn_plot.pack(fill="x", pady=10)

# Frame 3: Extrapolated Servo Configuration Array Reads Output
frame_results = tk.LabelFrame(left_control_panel, text=" 3. Calculated Servo Angle Values ", padx=10, pady=10, font=("Arial", 10, "bold"))
frame_results.pack(fill="x", pady=5)

label_result_a1 = tk.Label(frame_results, text="Angle 1 (Shoulder): --", font=("Arial", 11, "bold"), fg="#2c3e50")
label_result_a1.pack(anchor="w", pady=2)

label_result_a2 = tk.Label(frame_results, text="Angle 2 (Elbow): --", font=("Arial", 11, "bold"), fg="#2c3e50")
label_result_a2.pack(anchor="w", pady=2)

label_result_a3 = tk.Label(frame_results, text="Angle 3 (Wrist): --", font=("Arial", 11, "bold"), fg="#2c3e50")
label_result_a3.pack(anchor="w", pady=2)


# --- RIGHT PANEL CONFIGURATION ---
fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_aspect('equal', adjustable='box')
ax.set_title("3-DOF Planar Robot Arm Real-time Simulation Mesh", fontsize=10, fontweight='bold')

canvas = FigureCanvasTkAgg(fig, master=right_simulation_panel)
canvas.get_tk_widget().pack(fill="both", expand=True)

root.mainloop()