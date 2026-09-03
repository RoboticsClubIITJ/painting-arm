import os
import math
# import cv # REMOVE THIS LATER
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
# import Dynamixel.gui as gui
pen_controller = NanoPenController()
# --- Configuration ---
DXL_IDs = [0, 1, 2] # Shoulder (XM540), Elbow (XM430), Wrist (XM430)
BAUDRATE = 1000000
DEVICENAME = 'COM3'
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

def set_torque(enable):
    for dxl_id in DXL_IDs:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1 if enable else 0)

def set_operating_mode(mode):
    for dxl_id in DXL_IDs:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OPERATING_MODE, mode)

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
    
    for q_frame in trajectory_q:
        for i, dxl_id in enumerate(DXL_IDs):
            goal_tick = radians_to_ticks(q_frame[i], zero_ticks[i], invert=JOINT_INVERT[i])
            
            safe_max = max(max_ticks[i], min_ticks[i])
            safe_min = min(max_ticks[i], min_ticks[i])
            clamped_tick = max(min(goal_tick, safe_max), safe_min)

            write_signed_position(dxl_id, clamped_tick)
            
        time.sleep(1.0 / FPS)
        
    print(f"Finished drawing. End angles (rad): {np.round(trajectory_q[-1], 3)}")
    return trajectory_q[-1]

# --- 3. Interactive Command Loop ---
current_joint_angles = [0.0, 0.0, 0.0]

while True:
    try:
        cmd = input("\nCommand ('z'=zero, 'g'=go to XY, 't'=test path, 'q'=quit): ").strip().lower()
        
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
            print("Processing image...")
            raw_image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
            
            if raw_image is None:
                print("Image not found! Check the file path.")
            else:
                cv_paths = image_to_robot_paths(raw_image)
                print(f"Drawing {len(cv_paths)} separate strokes...")
                
                for i, path in enumerate(cv_paths):
                    if i == 0: continue
                    print(f"Executing stroke {i+1}/{len(cv_paths)}...")
                    
                    # Travel to the starting coordinate of the new stroke
                    start_x, start_y = path[0]
                    pen_controller.pen_up()
                    time.sleep(1)
                    current_joint_angles = go_to_xy_3dof(start_x, start_y, current_joint_angles)
                    time.sleep(1)
                    pen_controller.pen_down()
                    time.sleep(1)
                    
                    # Trace the contour smoothly
                    current_joint_angles = follow_path(path, current_joint_angles)

    except (KeyboardInterrupt, Exception) as e:
        pen_controller.pen_up()
        time.sleep(1)
        for i, dxl_id in enumerate(DXL_IDs):
            write_signed_position(dxl_id, zero_ticks[i])
        current_joint_angles = [0.0, 0.0, 0.0]
        print("Returned to ZERO position.")
        time.sleep(2)
        set_torque(False)
        print(f"\nTorque disabled. Exiting due to {e}.")
        break
