import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2

# ==========================================================
# === PHYSICAL PARAMETERS & LIMITS ===
# ==========================================================
L1 = 15.0
L2 = 12.0
L3 = 5.0
L_TUPLE = (L1, L2, L3)
MAX_REACH = L1 + L2 + L3

JOINT_LIMITS = [
    (np.radians(-150), np.radians(150)),  
    (np.radians(-150), np.radians(150)),  
    (np.radians(-150), np.radians(150))   
]

MAX_LINEAR_SPEED = 8.0  # cm/s (Task Space Speed)
FPS = 30
DT = 1.0 / FPS

# ==========================================================
# === KINEMATICS & FAST DLS SOLVER ===
# ==========================================================
def forward_kinematics_2d(q, L=L_TUPLE):
    t1, t2, t3 = q[0], q[1], q[2]
    t12 = t1 + t2
    t123 = t1 + t2 + t3
    x = L[0]*np.cos(t1) + L[1]*np.cos(t12) + L[2]*np.cos(t123)
    y = L[0]*np.sin(t1) + L[1]*np.sin(t12) + L[2]*np.sin(t123)
    return x, y

def ik_fast_dls(target_x, target_y, q_init, L=L_TUPLE, max_iter=100, tol=1e-2):
    dist = np.hypot(target_x, target_y)
    if dist > MAX_REACH:
        target_x *= (MAX_REACH - 0.01) / dist
        target_y *= (MAX_REACH - 0.01) / dist

    q = np.copy(q_init)
    I_2 = np.identity(2)
    damping = 0.05 
    
    for iteration in range(max_iter):
        t1, t2, t3 = q[0], q[1], q[2]
        t12, t123 = t1 + t2, t1 + t2 + t3
        
        curr_x = L[0]*np.cos(t1) + L[1]*np.cos(t12) + L[2]*np.cos(t123)
        curr_y = L[0]*np.sin(t1) + L[1]*np.sin(t12) + L[2]*np.sin(t123)
        
        error = np.array([target_x - curr_x, target_y - curr_y])
        if np.linalg.norm(error) < tol:
            return q, True, iteration + 1
            
        s1, s12, s123 = np.sin(t1), np.sin(t12), np.sin(t123)
        c1, c12, c123 = np.cos(t1), np.cos(t12), np.cos(t123)
        
        J = np.array([
            [-L[0]*s1 - L[1]*s12 - L[2]*s123, -L[1]*s12 - L[2]*s123, -L[2]*s123],
            [ L[0]*c1 + L[1]*c12 + L[2]*c123,  L[1]*c12 + L[2]*c123,  L[2]*c123]
        ])
        
        J_damp = J @ J.T + (damping**2) * I_2
        J_inv = J.T @ np.linalg.solve(J_damp, I_2)
        dq = J_inv @ error
        
        dq_norm = np.linalg.norm(dq)
        if dq_norm < 1e-4: break
            
        max_step = 0.30 
        if dq_norm > max_step: dq = dq * (max_step / dq_norm)
            
        q += dq
        q = (q + np.pi) % (2 * np.pi) - np.pi
        q[0] = np.clip(q[0], JOINT_LIMITS[0][0], JOINT_LIMITS[0][1])
        q[1] = np.clip(q[1], JOINT_LIMITS[1][0], JOINT_LIMITS[1][1])
        q[2] = np.clip(q[2], JOINT_LIMITS[2][0], JOINT_LIMITS[2][1])
        
    return q, False, max_iter

# ==========================================================
# === TASK SPACE TRAJECTORY GENERATOR (WITH VELOCITY PROFILES) ===
# ==========================================================
def generate_task_space_trajectory(x_start, y_start, x_end, y_end, q_start, v_max=MAX_LINEAR_SPEED, profile_type="Trapezoidal"):
    D_cart = np.hypot(x_end - x_start, y_end - y_start)
    
    if D_cart < 1e-4:
        q_end, _, _ = ik_fast_dls(x_end, y_end, q_init=q_start)
        return [q_end]
        
    T_total = max(DT, D_cart / v_max)
    num_steps = max(3, int(np.ceil(T_total / DT)))
    
    trajectory_q = []
    q_curr = np.copy(q_start)
    
    for i in range(num_steps):
        u = i / (num_steps - 1)
        
        # Apply Task-Space Velocity Profile Displacement s(u) in [0, 1]
        if profile_type == "S-Curve (Quintic)":
            s = 10 * (u**3) - 15 * (u**4) + 6 * (u**5)
        elif profile_type == "Trapezoidal":
            if u <= 1/3:
                s = 1.5 * (u**2)
            elif u <= 2/3:
                s = u - 1/6
            else:
                s = 1.0 - 1.5 * ((1.0 - u)**2)
        else:  # Linear
            s = u
            
        # Cartesian Straight-Line Interpolation
        x_curr = x_start + s * (x_end - x_start)
        y_curr = y_start + s * (y_end - y_start)
        
        # Solve Inverse Kinematics for Cartesian point
        q_next, _, _ = ik_fast_dls(x_curr, y_curr, q_init=q_curr)
        q_curr = q_next
        trajectory_q.append(np.copy(q_curr))
        
    return trajectory_q

# ==========================================================
# === GUI & CV APPLICATION ===
# ==========================================================
class Planar3DOFSimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3-DOF Drawing Robot — Task Space Velocity Profiling")
        self.q_current = np.array([np.pi/4, -np.pi/2, np.pi/4], dtype=float)
        
        self.traj_q = []
        self.traj_dq = []
        self.traj_draw_flags = []
        self.traj_idx = 0
        self.is_running = False
        self.stroke_index = 0

        self.setup_gui()
        self.setup_plots()

    def setup_gui(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(control_frame, text="Robot Controls", font=("Arial", 12, "bold")).pack(pady=(0, 10))

        ttk.Button(control_frame, text="Upload Image & Run", command=self.process_image).pack(pady=10, fill=tk.X)
        ttk.Button(control_frame, text="Reset Simulation", command=self.reset_sim).pack(pady=5, fill=tk.X)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=15)

        # Velocity Profile Dropdown
        ttk.Label(control_frame, text="Velocity Profile:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        self.profile_var = tk.StringVar(value="Trapezoidal")
        profile_dropdown = ttk.Combobox(
            control_frame, 
            textvariable=self.profile_var, 
            values=["Trapezoidal", "S-Curve (Quintic)", "Linear"],
            state="readonly"
        )
        profile_dropdown.pack(fill=tk.X, pady=(2, 15))

        # Animation Speed Slider
        ttk.Label(control_frame, text="Animation Speed:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        self.speed_var = tk.IntVar(value=1)
        speed_scale = ttk.Scale(control_frame, from_=1, to=10, variable=self.speed_var, orient=tk.HORIZONTAL)
        speed_scale.pack(fill=tk.X, pady=(2, 2))
        
        self.speed_label = ttk.Label(control_frame, text="1x Speed")
        self.speed_label.pack(anchor=tk.E)
        speed_scale.config(command=lambda v: self.speed_label.config(text=f"{int(float(v))}x Speed"))

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=15)

        self.status_var = tk.StringVar(value="STATUS: IDLE")
        ttk.Label(control_frame, textvariable=self.status_var, font=("Arial", 10, "bold"), wraplength=140).pack(pady=10)

    def setup_plots(self):
        self.fig, (self.ax_arm, self.ax_vel) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.ax_arm.set_title("Task Space Cartesian Path", fontweight='bold')
        self.ax_arm.set_aspect('equal')
        self.ax_arm.set_xlim(-5, MAX_REACH + 5)
        self.ax_arm.set_ylim(-15, 15)
        self.ax_arm.grid(True, linestyle='--')
        
        self.arm_line, = self.ax_arm.plot([], [], 'o-', lw=6, color='#2c3e50', zorder=4)
        self.brush_line, = self.ax_arm.plot([], [], 'o-', lw=4, color='#e74c3c', zorder=5)
        
        self.preview_lines = []
        self.trail_lines = []
        self.current_stroke_x = []
        self.current_stroke_y = []

        self.ax_vel.set_title("Task Space Driven Joint Velocities", fontweight='bold')
        self.ax_vel.set_xlabel("Time Step (sec)")
        self.ax_vel.set_ylabel("Velocity (rad/s)")
        self.ax_vel.grid(True, linestyle='--')
        self.v_lines = [self.ax_vel.plot([], [], label=l, lw=2)[0] for l in ['Shoulder', 'Elbow', 'Wrist']]
        self.ax_vel.legend(loc='upper right')

        self.draw_arm()

    def reset_sim(self):
        self.is_running = False
        self.traj_q = []; self.traj_dq = []; self.traj_draw_flags = []
        self.traj_idx = 0
        self.stroke_index = 0 
        
        for p in self.preview_lines: p.remove()
        self.preview_lines.clear()
        
        for t in self.trail_lines: t.remove()
        self.trail_lines.clear()
        
        self.current_stroke_x = []
        self.current_stroke_y = []
        for v in self.v_lines: v.set_data([], [])
        
        self.status_var.set("STATUS: RESET")
        self.draw_arm()

    def process_image(self):
        if self.is_running: return
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not filepath: return
        
        self.reset_sim()
        self.status_var.set("STATUS: PROCESSING CV...")
        self.root.update()

        # Step 1: Grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        cv2.imshow("Step 1: Grayscale (Press ENTER to continue)", img)
        cv2.waitKey(0) 

        # Step 2: Smoothing
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imshow("Step 2: Smoothed / Blurred (Press ENTER to continue)", blurred)
        cv2.waitKey(0)

        # Step 3: Canny Edge Detection
        edges = cv2.Canny(blurred, threshold1=120, threshold2=180)
        cv2.imshow("Step 3: Canny Edges (Press ENTER to continue)", edges)
        cv2.waitKey(0)

        # Step 4: Contours & Waypoints
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        preview_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        
        paths = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.003 * cv2.arcLength(cnt, True), True).reshape(-1, 2)
            if len(approx) > 2:
                x, y, w, h = cv2.boundingRect(approx)
                paths.append({'points': approx, 'y': y, 'x': x})
                
                cv2.polylines(preview_img, [approx], isClosed=True, color=(0, 255, 0), thickness=1)
                for pt in approx:
                    cv2.circle(preview_img, tuple(pt), 2, (0, 0, 255), -1)

        cv2.imshow("Step 4: Waypoint Extraction (Press ENTER to run simulation)", preview_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        paths.sort(key=lambda p: (p['y'], p['x']))
        h, w = img.shape
        
        # Scaling to Robot Workspace
        TARGET_W = 20.0 
        TARGET_H = 20.0 
        
        scale = min(TARGET_W / w, TARGET_H / h)
        center_x = 10.0 + (TARGET_W / 2.0)  
        center_y = 0 

        cv_paths = []
        for p_dict in paths:
            p = p_dict['points'].astype(float)
            mapped = []
            for point in p:
                cx_img = point[0] - (w / 2.0)
                cy_img = point[1] - (h / 2.0)
                
                mx = center_x + (cx_img * scale)
                my = center_y - (cy_img * scale) 
                mapped.append((mx, my))
            
            mapped.append(mapped[0]) # Close polygon
            cv_paths.append(mapped)
            
            p_arr = np.array(mapped)
            l, = self.ax_arm.plot(p_arr[:, 0], p_arr[:, 1], color='gray', linestyle='-', alpha=0.6)
            self.preview_lines.append(l)
            
        self.canvas.draw_idle()
        self.generate_full_trajectory(cv_paths)

    def generate_full_trajectory(self, cv_paths):
        profile = self.profile_var.get()
        self.status_var.set(f"STATUS: PLANNING ({profile.upper()})...")
        self.root.update()
        
        q_sim = self.q_current.copy()
        
        for path in cv_paths:
            curr_x, curr_y = forward_kinematics_2d(q_sim)
            
            # Transition Move (Pen Up)
            start_x, start_y = path[0]
            travel_traj = generate_task_space_trajectory(
                curr_x, curr_y, start_x, start_y, q_sim, 
                v_max=MAX_LINEAR_SPEED * 1.5, profile_type=profile
            )
            
            self.traj_q.extend(travel_traj)
            self.traj_draw_flags.extend([False] * len(travel_traj))
            q_sim = travel_traj[-1]
            
            # Drawing Stroke Moves (Pen Down)
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i+1]
                
                draw_traj = generate_task_space_trajectory(
                    x1, y1, x2, y2, q_sim, 
                    v_max=MAX_LINEAR_SPEED, profile_type=profile
                )
                
                self.traj_q.extend(draw_traj)
                self.traj_draw_flags.extend([True] * len(draw_traj))
                q_sim = draw_traj[-1]
        
        if len(self.traj_q) == 0:
            self.status_var.set("STATUS: NO VALID PATHS")
            return

        self.traj_q = np.array(self.traj_q)
        self.traj_dq = np.gradient(self.traj_q, DT, axis=0)
        self.ax_vel.set_xlim(0, len(self.traj_q) * DT)
        
        max_vel = np.max(np.abs(self.traj_dq))
        self.ax_vel.set_ylim(-max_vel * 1.2, max_vel * 1.2)
        
        self.is_running = True
        self.status_var.set("STATUS: EXECUTING...")
        self.run_animation_loop()

    def draw_arm(self):
        t1, t2, t3 = self.q_current
        p1 = np.array([L1 * np.cos(t1), L1 * np.sin(t1)])
        p2 = p1 + np.array([L2 * np.cos(t1+t2), L2 * np.sin(t1+t2)])
        p3 = p2 + np.array([L3 * np.cos(t1+t2+t3), L3 * np.sin(t1+t2+t3)])

        self.arm_line.set_data([0, p1[0], p2[0]], [0, p1[1], p2[1]])
        self.brush_line.set_data([p2[0], p3[0]], [p2[1], p3[1]])
        self.canvas.draw_idle()
        return p3

    def run_animation_loop(self):
        if self.is_running and self.traj_idx < len(self.traj_q):
            # Apply dynamic animation speed step multiplier
            step = int(self.speed_var.get())
            
            self.q_current = self.traj_q[self.traj_idx]
            is_drawing = self.traj_draw_flags[self.traj_idx]
            
            p3 = self.draw_arm()
            
            if is_drawing:
                self.current_stroke_x.append(p3[0])
                self.current_stroke_y.append(p3[1])
                
                if len(self.trail_lines) == self.stroke_index:
                    new_trail, = self.ax_arm.plot([], [], 'b-', lw=1.5, zorder=2)
                    self.trail_lines.append(new_trail)
                    
                self.trail_lines[self.stroke_index].set_data(self.current_stroke_x, self.current_stroke_y)
                
            else:
                if len(self.current_stroke_x) > 0:
                    self.current_stroke_x = []
                    self.current_stroke_y = []
                    self.stroke_index += 1

            t_arr = np.arange(self.traj_idx + 1) * DT
            for j in range(3):
                self.v_lines[j].set_data(t_arr, self.traj_dq[:self.traj_idx + 1, j])
            
            self.traj_idx += step
            
            # Schedule next frame update
            self.root.after(int(DT * 1000 / step), self.run_animation_loop)
            
        elif self.is_running:
            self.is_running = False
            self.status_var.set("STATUS: DRAWING COMPLETE")

if __name__ == '__main__':
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", lambda: (root.destroy(), plt.close('all')))
    app = Planar3DOFSimApp(root)
    root.mainloop()
