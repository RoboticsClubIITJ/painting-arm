import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
from scipy.interpolate import splprep, splev
import scipy.signal as signal

# ==========================================================
# === PHYSICAL PARAMETERS & LIMITS ===
# ==========================================================
L1 = 15.0
L2 = 12.0
L3 = 5.0
L_TUPLE = (L1, L2, L3)
MAX_REACH = L1 + L2 + L3

JOINT_LIMITS = [
    (np.radians(-120), np.radians(120)),  
    (np.radians(-150), np.radians(150)),  
    (np.radians(-150), np.radians(150))   
]

MAX_LINEAR_SPEED = 15.0 # cm/s (Desired Task Space Speed)

# PHYSICAL SERVO LIMITS
MAX_JOINT_VEL = 1.0     # rad/s 
MAX_JOINT_ACC = 0.2     # rad/s^2 
FPS = 30
DT = 1.0 / FPS
REVIEW_WINDOW = 5.0 # Fixed 5-second viewing window

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
# === ADVANCED CONTINUOUS PATH WITH KINEMATIC LIMITS ===
# ==========================================================
def get_interpolated_point(points, cum_dist, current_segment, target_dist):
    seg_start = cum_dist[current_segment]
    seg_end = cum_dist[current_segment + 1]
    seg_len = seg_end - seg_start
    
    if seg_len > 1e-6:
        ratio = (target_dist - seg_start) / seg_len
    else:
        ratio = 1.0
        
    ratio = max(0.0, min(1.0, ratio))
    p0 = points[current_segment]
    p1 = points[current_segment + 1]
    return p0 + ratio * (p1 - p0)

def generate_continuous_trajectory(points, q_start, v_max=MAX_LINEAR_SPEED):
    if len(points) < 2: return []

    points = np.array(points)
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(segment_lengths), 0, 0)
    total_dist = cum_dist[-1]

    if total_dist < 1e-4:
        q_end, _, _ = ik_fast_dls(points[-1, 0], points[-1, 1], q_init=q_start)
        return [q_end]

    trajectory_q = []
    q_curr = np.copy(q_start)
    dq_curr = np.zeros(3) 
    
    dist = 0.0
    current_segment = 0

    while dist < total_dist:
        dist_left = total_dist - dist
        
        # Step A: Velocity profile with Look-Ahead Deceleration
        s_val = dist / total_dist
        v_desired = v_max * (16 * s_val**2 * (1 - s_val)**2)
        
        safe_decel_rate = 3.5 
        v_brake = np.sqrt(2 * safe_decel_rate * max(0.0, dist_left))
        
        v_desired = min(v_desired, v_brake)
        
        if dist_left < 0.1:
            v_desired = max(v_desired, 0.01)
        else:
            v_desired = max(v_desired, 0.05 * v_max)

        # Step B: Find path tangent direction (u_hat)
        p0 = points[current_segment]
        p1 = points[current_segment + 1]
        vec = p1 - p0
        norm_vec = np.linalg.norm(vec)
        u_hat = (vec / norm_vec) if norm_vec > 1e-6 else np.array([1.0, 0.0])

        # Step C: Calculate dynamic Jacobian mapping at current pose
        t1, t2, t3 = q_curr
        s1, s12, s123 = np.sin(t1), np.sin(t1+t2), np.sin(t1+t2+t3)
        c1, c12, c123 = np.cos(t1), np.cos(t1+t2), np.cos(t1+t2+t3)
        
        J = np.array([
            [-L1*s1 - L2*s12 - L3*s123, -L2*s12 - L3*s123, -L3*s123],
            [ L1*c1 + L2*c12 + L3*c123,  L2*c12 + L3*c123,  L3*c123]
        ])
        
        I_2 = np.identity(2)
        damping = 0.05
        J_damp = J @ J.T + (damping**2) * I_2
        J_inv = J.T @ np.linalg.solve(J_damp, I_2)
        
        L_vec = J_inv @ u_hat 

        # Step D: ENFORCE PHYSICAL SERVO LIMITS
        v_vlim = min([MAX_JOINT_VEL / abs(l) if abs(l) > 1e-6 else float('inf') for l in L_vec])
        
        v_max_allowed = float('inf')
        v_min_allowed = 0.0
        
        for i in range(3):
            l = L_vec[i]
            bound_upper = dq_curr[i] + MAX_JOINT_ACC * DT
            bound_lower = dq_curr[i] - MAX_JOINT_ACC * DT
            
            if l > 1e-6:
                v_max_i = bound_upper / l
                v_min_i = bound_lower / l
            elif l < -1e-6:
                v_max_i = bound_lower / l
                v_min_i = bound_upper / l
            else:
                v_max_i = float('inf')
                v_min_i = -float('inf')
                
            if v_max_i < v_max_allowed: v_max_allowed = v_max_i
            if v_min_i > v_min_allowed: v_min_allowed = v_min_i

        if v_min_allowed > v_max_allowed:
            v_min_allowed = v_max_allowed

        v_actual = min(v_desired, v_vlim, v_max_allowed)
        v_actual = max(v_actual, v_min_allowed)
        
        # Maintain a reasonable minimum speed (0.5 cm/s) so it doesn't freeze at the end of lines
        v_actual = max(v_actual, 0.5) 

        # Step E: Move forward along the path
        step_dist = v_actual * DT
        if dist + step_dist > total_dist:
            step_dist = total_dist - dist
            
        dist += step_dist
        
        while current_segment < len(cum_dist) - 2 and dist > cum_dist[current_segment + 1]:
            current_segment += 1
            
        p_target = get_interpolated_point(points, cum_dist, current_segment, dist)
        q_next, _, _ = ik_fast_dls(p_target[0], p_target[1], q_init=q_curr)
        
        dq_curr = (q_next - q_curr) / DT
        q_curr = q_next
        
        trajectory_q.append(np.copy(q_curr))

    return trajectory_q

# ==========================================================
# === GUI & CV APPLICATION ===
# ==========================================================
class Planar3DOFSimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3-DOF Drawing (Kinematic Limits + Face Portrait Mode)")
        self.q_current = np.array([np.pi/4, -np.pi/2, np.pi/4], dtype=float)
        
        self.raw_image = None
        self.final_cv_paths = []

        self.traj_q = []
        self.traj_dq = []
        self.traj_draw_flags = []
        self.traj_pen_status = []
        self.traj_idx = 0
        self.is_running = False
        self.stroke_index = 0

        self.setup_gui()
        self.setup_plots()

        # Bind the Enter key to update the preview
        self.root.bind('<Return>', self.update_preview)

    def setup_gui(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(control_frame, text="Upload Image", command=self.upload_image).pack(pady=10, fill=tk.X)
        
        # --- Canny Threshold Sliders with Dynamic Labels ---
        thresh_frame = ttk.LabelFrame(control_frame, text="Canny Edge Thresholds")
        thresh_frame.pack(pady=10, fill=tk.X)
        
        self.t1_var = tk.IntVar(value=60)
        self.lbl_t1 = ttk.Label(thresh_frame, text=f"Threshold 1 (Min): {self.t1_var.get()}")
        self.lbl_t1.pack()
        ttk.Scale(thresh_frame, from_=0, to=255, variable=self.t1_var, orient=tk.HORIZONTAL, 
                  command=lambda v: self.lbl_t1.config(text=f"Threshold 1 (Min): {int(float(v))}")).pack(fill=tk.X, padx=5)
        
        self.t2_var = tk.IntVar(value=140)
        self.lbl_t2 = ttk.Label(thresh_frame, text=f"Threshold 2 (Max): {self.t2_var.get()}")
        self.lbl_t2.pack()
        ttk.Scale(thresh_frame, from_=0, to=255, variable=self.t2_var, orient=tk.HORIZONTAL, 
                  command=lambda v: self.lbl_t2.config(text=f"Threshold 2 (Max): {int(float(v))}")).pack(fill=tk.X, padx=5)

        ttk.Button(thresh_frame, text="Update Preview (Enter)", command=self.update_preview).pack(pady=10, fill=tk.X)

        self.btn_start = ttk.Button(control_frame, text="Start Draw", command=self.start_drawing, state=tk.DISABLED)
        self.btn_start.pack(pady=15, fill=tk.X)

        ttk.Button(control_frame, text="Reset", command=self.reset_sim).pack(pady=5, fill=tk.X)

        # --- Timeline Review Slider ---
        review_frame = ttk.LabelFrame(control_frame, text="Timeline Review (Post-Draw)")
        review_frame.pack(pady=15, fill=tk.X)
        
        self.timeline_var = tk.DoubleVar(value=0.0)
        self.timeline_slider = ttk.Scale(
            review_frame, from_=0.0, to=5.0, 
            variable=self.timeline_var, orient=tk.HORIZONTAL,
            command=self.on_scroll_timeline
        )
        self.timeline_slider.pack(fill=tk.X, padx=5, pady=5)
        self.timeline_slider.state(['disabled'])
        
        self.timeline_label = ttk.Label(review_frame, text="Window: 0.0s - 5.0s")
        self.timeline_label.pack(pady=(0, 5))

        self.status_var = tk.StringVar(value="STATUS: IDLE")
        ttk.Label(control_frame, textvariable=self.status_var, font=("Arial", 10, "bold")).pack(pady=10)

    def on_scroll_timeline(self, val):
        if self.is_running:
            return
            
        t_start = float(val)
        t_end = t_start + REVIEW_WINDOW
        
        self.timeline_label.config(text=f"Window: {t_start:.1f}s - {t_end:.1f}s")
        for ax in self.vel_axes:
            ax.set_xlim(t_start, t_end)
            
        self.canvas.draw_idle()

    def setup_plots(self):
        self.fig = plt.figure(figsize=(14, 7), dpi=100)
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.2, 1], hspace=0.4)
        
        self.ax_arm = self.fig.add_subplot(gs[:, 0])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.ax_arm.set_title("Task Space Cartesian Paths", fontweight='bold')
        self.ax_arm.set_aspect('equal')
        self.ax_arm.set_xlim(-5, MAX_REACH + 5)
        self.ax_arm.set_ylim(-15, 15)
        self.ax_arm.grid(True, linestyle='--')
        
        self.arm_line, = self.ax_arm.plot([], [], 'o-', lw=6, color='#2c3e50', zorder=4)
        self.brush_line, = self.ax_arm.plot([], [], 'o-', lw=4, color='#e74c3c', zorder=5)
        
        self.pen_status_text = self.ax_arm.text(0.03, 0.95, '', transform=self.ax_arm.transAxes, 
                                                fontsize=11, fontweight='bold', color='#c0392b',
                                                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

        self.preview_lines = []
        self.trail_lines = []
        self.current_stroke_x = []
        self.current_stroke_y = []

        self.vel_axes = [
            self.fig.add_subplot(gs[0, 1]),
            self.fig.add_subplot(gs[1, 1]),
            self.fig.add_subplot(gs[2, 1])
        ]
        
        joint_names = ['Shoulder', 'Elbow', 'Wrist']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        self.v_lines = []
        
        for i, ax in enumerate(self.vel_axes):
            ax.set_title(f"{joint_names[i]} Velocity (Limit: {MAX_JOINT_VEL} rad/s)", fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--')
            
            # Set the fixed Y-axis bounds directly
            ax.set_ylim(-0.4, 0.4)
            
            ax.axhline(MAX_JOINT_VEL, color='red', linestyle=':', alpha=0.5)
            ax.axhline(-MAX_JOINT_VEL, color='red', linestyle=':', alpha=0.5)

            line, = ax.plot([], [], lw=2, color=colors[i])
            self.v_lines.append(line)
            
        self.vel_axes[-1].set_xlabel("Time (sec)")
        
        self.draw_arm()

    def reset_sim(self):
        self.is_running = False
        self.traj_q = []; self.traj_dq = []; self.traj_draw_flags = []; self.traj_pen_status = []
        self.traj_idx = 0
        self.stroke_index = 0 
        self.raw_image = None
        self.final_cv_paths = []
        
        for p in self.preview_lines: p.remove()
        self.preview_lines.clear()
        
        for t in self.trail_lines: t.remove()
        self.trail_lines.clear()
        
        self.current_stroke_x = []
        self.current_stroke_y = []
        for v in self.v_lines: v.set_data([], [])
        
        self.timeline_slider.state(['disabled'])
        self.timeline_var.set(0.0)
        self.timeline_label.config(text="Window: 0.0s - 5.0s")
        
        self.btn_start.config(state=tk.DISABLED)
        self.pen_status_text.set_text('')
        self.status_var.set("STATUS: RESET")
        self.draw_arm()

    def upload_image(self):
        if self.is_running: return
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not filepath: return
        
        self.reset_sim()
        self.raw_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        self.status_var.set("STATUS: IMAGE UPLOADED. ADJUST SLIDERS & ENTER.")
        self.update_preview()

    def update_preview(self, event=None):
        if self.raw_image is None or self.is_running: return
        
        self.status_var.set("STATUS: UPDATING PREVIEW...")
        self.root.update()

        # 1. PORTRAIT PRE-PROCESSING
        face_smoothed = cv2.bilateralFilter(self.raw_image, d=9, sigmaColor=75, sigmaSpace=75)
        
        # 2. EDGE DETECTION (Using Sliders)
        t1 = self.t1_var.get()
        t2 = self.t2_var.get()
        edges = cv2.Canny(face_smoothed, threshold1=t1, threshold2=t2)

        # 3. MORPHOLOGICAL CLOSE
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 4. CONTOUR DETECTION
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        paths = []
        
        for cnt in contours:
            cnt = cnt.squeeze()
            if cnt.ndim == 1:
                continue
                
            if cv2.arcLength(cnt, closed=False) < 30.0:
                continue

            diffs = np.diff(cnt, axis=0)
            mask = np.any(diffs != 0, axis=1)
            cnt = cnt[np.append([True], mask)]
            
            if len(cnt) < 5:
                continue

            # 5. SPLINE INTERPOLATION 
            x = cnt[:, 0]
            y = cnt[:, 1]
            
            try:
                tck, u = splprep([x, y], s=3.0, per=True) 
                
                arc_len = cv2.arcLength(cnt, closed=False)
                num_points = max(10, int(arc_len * 0.5)) 
                
                u_new = np.linspace(u.min(), u.max(), num_points)
                x_new, y_new = splev(u_new, tck)
                
                smooth_approx = np.vstack((x_new, y_new)).T
            except:
                smooth_approx = cnt 

            x_bound, y_bound, w, h = cv2.boundingRect(smooth_approx.astype(np.float32))
            paths.append({'points': smooth_approx, 'y': y_bound, 'x': x_bound})

        # Clear old preview lines
        for p in self.preview_lines:
            p.remove()
        self.preview_lines.clear()

        # Scale and map to the robot's physical workspace
        paths.sort(key=lambda p: (p['y'], p['x']))
        h, w = self.raw_image.shape
        TARGET_W, TARGET_H = 20.0, 20.0 
        scale = min(TARGET_W / w, TARGET_H / h)
        center_x, center_y = 10.0 + (TARGET_W / 2.0), 0 

        self.final_cv_paths = []
        for p_dict in paths:
            p = p_dict['points'].astype(float)
            mapped = []
            for point in p:
                cx_img, cy_img = point[0] - (w / 2.0), point[1] - (h / 2.0)
                mapped.append((center_x + (cx_img * scale), center_y - (cy_img * scale)))
            
            mapped.append(mapped[0])
            self.final_cv_paths.append(mapped)
            
            p_arr = np.array(mapped)
            l, = self.ax_arm.plot(p_arr[:, 0], p_arr[:, 1], color='gray', linestyle='-', alpha=0.6)
            self.preview_lines.append(l)
            
        self.canvas.draw_idle()
        self.status_var.set("STATUS: PREVIEW READY. PRESS START.")
        self.btn_start.config(state=tk.NORMAL)

    def start_drawing(self):
        if not self.final_cv_paths: return
        self.btn_start.config(state=tk.DISABLED)
        self.generate_full_trajectory(self.final_cv_paths)

    def generate_full_trajectory(self, cv_paths):
        self.status_var.set("STATUS: PLANNING TASK SPACE PATH...")
        self.root.update()
        
        q_sim = self.q_current.copy()
        pause_frames = int(0.2 * FPS) # Fast pause (0.2 seconds)
        
        for i, path in enumerate(cv_paths):
            curr_x, curr_y = forward_kinematics_2d(q_sim)
            start_x, start_y = path[0]
            
            # --- Pen Up & Travel ---
            travel_traj = generate_continuous_trajectory([(curr_x, curr_y), (start_x, start_y)], q_sim, v_max=MAX_LINEAR_SPEED * 1.5)
            
            if travel_traj:
                # 1. Lift Pen Pause (Only if we just finished a previous shape)
                if i > 0:
                    self.traj_q.extend([q_sim] * pause_frames)
                    self.traj_draw_flags.extend([False] * pause_frames)
                    self.traj_pen_status.extend(["LIFTING PEN"] * pause_frames)

                # 2. Travel smoothly
                self.traj_q.extend(travel_traj)
                self.traj_draw_flags.extend([False] * len(travel_traj))
                self.traj_pen_status.extend(["PEN UP (TRAVEL)"] * len(travel_traj))
                q_sim = travel_traj[-1]
            
            # 3. Lower Pen Pause
            self.traj_q.extend([q_sim] * pause_frames)
            self.traj_draw_flags.extend([False] * pause_frames)
            self.traj_pen_status.extend(["LOWERING PEN"] * pause_frames)

            # --- Pen Down & Draw ---
            draw_traj = generate_continuous_trajectory(path, q_sim, v_max=MAX_LINEAR_SPEED)
            if draw_traj:
                self.traj_q.extend(draw_traj)
                self.traj_draw_flags.extend([True] * len(draw_traj))
                self.traj_pen_status.extend(["PEN DOWN (DRAWING)"] * len(draw_traj))
                q_sim = draw_traj[-1]
        
        if not self.traj_q:
            self.status_var.set("STATUS: NO VALID PATHS"); return

        self.traj_q = np.array(self.traj_q)
        
        # Savitzky-Golay trajectory smoothing to eliminate any remaining micro-vibrations
        window_length = 15  
        if len(self.traj_q) > window_length:
            for joint_idx in range(3):
                self.traj_q[:, joint_idx] = signal.savgol_filter(
                    self.traj_q[:, joint_idx], 
                    window_length=window_length, 
                    polyorder=3
                )

        self.traj_dq = np.gradient(self.traj_q, DT, axis=0)
        
        total_duration = len(self.traj_q) * DT
        max_start_time = max(0.0, total_duration - REVIEW_WINDOW)
        self.timeline_slider.config(to=max_start_time)
        
        for ax in self.vel_axes:
            ax.set_ylim(-0.8, 0.8)
        
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
            self.q_current = self.traj_q[self.traj_idx]
            is_drawing = self.traj_draw_flags[self.traj_idx]
            pen_status = self.traj_pen_status[self.traj_idx]
            p3 = self.draw_arm()

            self.pen_status_text.set_text(f"Z-AXIS: {pen_status}")
            
            if is_drawing:
                self.current_stroke_x.append(p3[0])
                self.current_stroke_y.append(p3[1])
                if len(self.trail_lines) == self.stroke_index:
                    new_trail, = self.ax_arm.plot([], [], 'b-', lw=1.5, zorder=2)
                    self.trail_lines.append(new_trail)
                self.trail_lines[self.stroke_index].set_data(self.current_stroke_x, self.current_stroke_y)
            else:
                if len(self.current_stroke_x) > 0:
                    self.current_stroke_x, self.current_stroke_y = [], []
                    self.stroke_index += 1

            t_arr = np.arange(self.traj_idx + 1) * DT
            current_time = t_arr[-1]
            
            t_min = max(0, current_time - REVIEW_WINDOW)
            t_max = max(REVIEW_WINDOW, current_time)

            for i, ax in enumerate(self.vel_axes):
                self.v_lines[i].set_data(t_arr, self.traj_dq[:self.traj_idx + 1, i])
                ax.set_xlim(t_min, t_max)
                
            self.timeline_var.set(t_min)
            self.timeline_label.config(text=f"Window: {t_min:.1f}s - {t_max:.1f}s")
            
            self.traj_idx += 1
            self.root.after(int(DT * 1000), self.run_animation_loop)
            
        elif self.is_running:
            self.is_running = False
            self.pen_status_text.set_text("DRAWING COMPLETE")
            self.status_var.set("STATUS: COMPLETE. USE SLIDER TO REVIEW.")
            self.timeline_slider.state(['!disabled']) 

if __name__ == '__main__':
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", lambda: (root.destroy(), plt.close('all')))
    app = Planar3DOFSimApp(root)
    root.mainloop()
