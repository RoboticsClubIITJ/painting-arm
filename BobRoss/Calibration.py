import os
import json
import math
# import cv2 # REMOVE THIS LATER
import numpy as np
import dynamixel_sdk as dxl
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Trajectory import generate_continuous_trajectory
from Configurations import MAX_LINEAR_SPEED, FPS
from Kinematics import ik_fast_dls
from Pen_up_down import *
from cv import *
from gui import run_gui


# import Dynamixel.gui as gui
pen_controller = NanoPenController()
# --- Configuration ---
DXL_IDs = [0, 1, 2] # Shoulder (XM540), Elbow (XM430), Wrist (XM430)
BAUDRATE = 1000000
DEVICENAME = '/dev/ttyUSB1'
PROTOCOL_VERSION = 2.0

ADDR_OPERATING_MODE = 11        # Operating Mode Address
EXT_POSITION_CONTROL_MODE = 4   # Value for Extended Position Control

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

# Invert flag for each joint [Shoulder, Elbow, Wrist].
# Set to True if increasing ticks produces CW rotation physically.
JOINT_INVERT = [True, True, True]

REL_LIMIT_1 = [-1575, -1456, -1340]
REL_LIMIT_2 = [1468, 1256, 1396]

portHandler = dxl.PortHandler(DEVICENAME)
packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)

# --- Stall / Power-Loss Detection ---
STALL_CHECK_INTERVAL = int(FPS * 0.5)   # check twice a second
STALL_POSITION_THRESHOLD = 5            # ticks - actual movement below this = "not moving"
STALL_COMMANDED_THRESHOLD = 15          # ticks - commanded movement above this = "should be moving"
ROLLBACK_STROKES_ON_STALL = 3  #how many strokes to subtract in log file

class ArmStallError(Exception):
    """Raised when servos are commanded to move but don't — signals power loss or mechanical failure."""
    pass

if not portHandler.openPort() or not portHandler.setBaudRate(BAUDRATE):
    print("Failed to open port or set baudrate. Check COM port and power.")
    quit()

# --- SDK Helper Functions ---
def read_signed_position(dxl_id):
    """Reads position and converts 32-bit unsigned to signed integer."""
    pos, _, _ = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_POSITION)
    if pos > 2147483647:
        pos -= 4294967296
    return pos

def write_signed_position(dxl_id, tick):
    """Converts signed integer to 32-bit unsigned for the Dynamixel SDK."""
    tick = int(tick)
    if tick < 0:
        tick += 4294967296
    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, tick)

def check_for_stall(prev_actual, prev_commanded, curr_commanded):
    """Compares actual servo movement vs commanded movement over the last check window."""
    actual = [read_signed_position(dxl_id) for dxl_id in DXL_IDs]
    for i in range(len(DXL_IDs)):
        commanded_delta = abs(curr_commanded[i] - prev_commanded[i])
        actual_delta = abs(actual[i] - prev_actual[i])
        if commanded_delta > STALL_COMMANDED_THRESHOLD and actual_delta < STALL_POSITION_THRESHOLD:
            raise ArmStallError(
                f"Joint {DXL_IDs[i]} not responding (commanded {commanded_delta} ticks, "
                f"moved {actual_delta} ticks). Likely power loss."
            )
    return actual

def set_torque(enable):
    for dxl_id in DXL_IDs:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1 if enable else 0)

def set_operating_mode(mode):
    for dxl_id in DXL_IDs:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OPERATING_MODE, mode)

# --- Progress Logging (for crash recovery) ---
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "current_drawing.json")

def save_new_session(zero_ticks_list, cv_paths):
    """Called the moment strokes are generated & sent, before drawing starts."""
    os.makedirs(LOG_DIR, exist_ok=True)
    strokes_serializable = [
        [[float(pt[0]), float(pt[1])] for pt in path] for path in cv_paths
    ]
    data = {
        "zero_ticks": [int(z) for z in zero_ticks_list],
        "strokes": strokes_serializable,
        "last_completed_stroke": -1,
    }
    with open(LOG_FILE, "w") as f:
        json.dump(data, f)

def update_progress(stroke_index):
    with open(LOG_FILE, "r") as f:
        data = json.load(f)
    data["last_completed_stroke"] = stroke_index
    with open(LOG_FILE, "w") as f:
        json.dump(data, f)

def load_session():
    with open(LOG_FILE, "r") as f:
        return json.load(f)

def rollback_progress(num_strokes):
    """Called when a stall is detected — the last few logged 'completed' strokes
    may actually be false/partial data written before the failure was caught."""
    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return

    data["last_completed_stroke"] = max(-1, data["last_completed_stroke"] - num_strokes)

    with open(LOG_FILE, "w") as f:
        json.dump(data, f)

    print(f"Rolled back progress by {num_strokes} strokes. "
          f"Log now shows last_completed_stroke = {data['last_completed_stroke']}.")

# --- 1. Initialization and Calibration Phase ---
set_torque(False)

print("\n--- CONFIGURING EXTENDED POSITION CONTROL ---")
set_operating_mode(EXT_POSITION_CONTROL_MODE)
print("Extended Position Control (Multi-turn) Enabled.")

print("\n--- ZERO CALIBRATION ---")
input("Move the ENTIRE arm straight along the X-axis (0 radians). Press Enter...")
zero_ticks = []
for dxl_id in DXL_IDs:
    zero_ticks.append(read_signed_position(dxl_id))
print(f"Zero offsets recorded: {zero_ticks}")

print("\n--- APPLYING RELATIVE SAFETY LIMITS ---")
max_ticks = []
min_ticks = []

for i, dxl_id in enumerate(DXL_IDs):
    limit_a = zero_ticks[i] + REL_LIMIT_1[i]
    limit_b = zero_ticks[i] + REL_LIMIT_2[i]
    
    max_ticks.append(limit_a)
    min_ticks.append(limit_b)
    print(f"Servo {dxl_id} limits set to: {limit_a} and {limit_b}")

time.sleep(2)
set_torque(True)
print("\nTorque ENABLED. Arm is locked and ready.")

# --- 2. Motion Helpers ---
def radians_to_ticks(radians, zero_offset, invert=False):
    direction = -1 if invert else 1
    ticks_offset = int(direction * radians * (4096 / (2 * math.pi)))
    return zero_offset + ticks_offset

def go_to_xy_3dof(target_x, target_y, current_angles):
    """Moves directly to a single target coordinate."""
    q_target, success, _ = ik_fast_dls(target_x, target_y, q_init=current_angles)
    
    if not success:
        print(f"Warning: Target ({target_x}, {target_y}) might be out of reach or near singularity.")

    for i, dxl_id in enumerate(DXL_IDs):
        goal_tick = radians_to_ticks(q_target[i], zero_ticks[i], invert=JOINT_INVERT[i])
        
        safe_max = max(max_ticks[i], min_ticks[i])
        safe_min = min(max_ticks[i], min_ticks[i])
        clamped_tick = max(min(goal_tick, safe_max), safe_min)

        if goal_tick != clamped_tick:
            print(f"Safety constraint triggered on Joint {dxl_id}.")

        write_signed_position(dxl_id, clamped_tick)
    
    print(f"Moved to X:{target_x}, Y:{target_y} | Angles (rad): {np.round(q_target, 3)}")
    return q_target

def follow_path(path_points, current_angles):
    """Traces a polyline array smoothly using the trajectory planner."""
    trajectory_q = generate_continuous_trajectory(path_points, current_angles, v_max=MAX_LINEAR_SPEED)
    
    if not trajectory_q:
        print("Path generation failed or path too short.")
        return current_angles
        
    print(f"Executing path with {len(trajectory_q)} frames...")
    
    prev_actual = [read_signed_position(dxl_id) for dxl_id in DXL_IDs]   
    prev_commanded = list(prev_actual)                                   
    frame_counter = 0 

    for q_frame in trajectory_q:
        clamped_ticks = [] 
        for i, dxl_id in enumerate(DXL_IDs):
            goal_tick = radians_to_ticks(q_frame[i], zero_ticks[i], invert=JOINT_INVERT[i])
            
            safe_max = max(max_ticks[i], min_ticks[i])
            safe_min = min(max_ticks[i], min_ticks[i])
            clamped_tick = max(min(goal_tick, safe_max), safe_min)
            clamped_ticks.append(clamped_tick) 

            write_signed_position(dxl_id, clamped_tick)
            
        time.sleep(1.0 / FPS)
        
        frame_counter += 1                                               
        if frame_counter >= STALL_CHECK_INTERVAL:                        
            prev_actual = check_for_stall(prev_actual, prev_commanded, clamped_ticks)  
            prev_commanded = clamped_ticks                               
            frame_counter = 0          

    print(f"Finished drawing. End angles (rad): {np.round(trajectory_q[-1], 3)}")
    # _ = go_to_xy_3dof(, path_points[-1][1], trajectory_q[-1]) # make the arm go to (0, -400)
    return trajectory_q[-1]

# --- 3. Interactive Command Loop ---
current_joint_angles = [0.0, 0.0, 0.0]

while True:
    try:
        cmd = input("\nCommand ('z'=zero, 'g'=go to XY, 't'=test path, 'r'=recover, 'q'=quit): ").strip().lower()
        
        if cmd == 'q':
            set_torque(False)
            print("Torque disabled. Exiting.")
            break
            
        elif cmd == 'z':
            for i, dxl_id in enumerate(DXL_IDs):
                write_signed_position(dxl_id, zero_ticks[i])
            current_joint_angles = [0.0, 0.0, 0.0]
            print("Returned to ZERO position.")
            
        elif cmd == 'g':
            try:
                tx = float(input("Enter target X (cm): "))
                ty = float(input("Enter target Y (cm): "))
                current_joint_angles = go_to_xy_3dof(tx, ty, current_joint_angles)
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                
        elif cmd == 't':
            print("Opening GUI for image processing...")
            cv_paths = run_gui()
            
            if not cv_paths:
                print("GUI closed without generating paths. Ensure an image was uploaded.")
            else:
                print(f"Drawing {len(cv_paths)} separate strokes...")
                save_new_session(zero_ticks, cv_paths) 

                for i, path in enumerate(cv_paths):
                    # if i == 0: continue
                    print(f"Executing stroke {i+1}/{len(cv_paths)}...")
                    
                    start_x, start_y = path[0]
                    pen_controller.pen_up()
                    time.sleep(1)
                    current_joint_angles = go_to_xy_3dof(start_x, start_y, current_joint_angles)
                    time.sleep(1)
                    pen_controller.pen_down()
                    time.sleep(1)
                    
                    # Trace the contour smoothly
                    current_joint_angles = follow_path(path, current_joint_angles)
                    update_progress(i) 
        
        elif cmd == 'r':
            print("Loading previous session from log file...")
            try:
                session = load_session()
            except FileNotFoundError:
                print(f"No log file found at {LOG_FILE}. Nothing to recover.")
                continue

            saved_zero = session["zero_ticks"]
            strokes = session["strokes"]
            last_completed = session["last_completed_stroke"]

            print(f"Recovered zero ticks: {saved_zero}")
            print(f"Resuming after stroke index {last_completed} (of {len(strokes)-1})")

            # Overwrite this session's calibrated zero with the recovered one
            zero_ticks = saved_zero
            # Recompute safety limits around the recovered zero so clamping stays consistent
            max_ticks = []
            min_ticks = []
            for i, dxl_id in enumerate(DXL_IDs):
                max_ticks.append(zero_ticks[i] + REL_LIMIT_1[i])
                min_ticks.append(zero_ticks[i] + REL_LIMIT_2[i])

            for i, dxl_id in enumerate(DXL_IDs):
                write_signed_position(dxl_id, zero_ticks[i])
            current_joint_angles = [0.0, 0.0, 0.0]
            time.sleep(1)

            for i in range(last_completed + 1, len(strokes)):
                path = strokes[i]
                print(f"Executing stroke {i+1}/{len(strokes)}...")

                start_x, start_y = path[0]
                pen_controller.pen_up()
                time.sleep(1)
                current_joint_angles = go_to_xy_3dof(start_x, start_y, current_joint_angles)
                time.sleep(1)
                pen_controller.pen_down()
                time.sleep(1)

                current_joint_angles = follow_path(path, current_joint_angles)
                update_progress(i)

            print("Recovery drawing complete.")


                
    except (KeyboardInterrupt, Exception) as e:
        pen_controller.pen_up()
        time.sleep(1)
        if isinstance(e, ArmStallError):        # <-- ADD
            rollback_progress(ROLLBACK_STROKES_ON_STALL)
        for i, dxl_id in enumerate(DXL_IDs):
            write_signed_position(dxl_id, zero_ticks[i])
        current_joint_angles = [0.0, 0.0, 0.0]
        print("Returned to ZERO position.")
        time.sleep(2)
        set_torque(False)
        print(f"\nTorque disabled. Exiting due to {e}.")
        break
