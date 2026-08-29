import os
import math
import numpy as np
import dynamixel_sdk as dxl
import time

# --- Configuration ---
DXL_IDs = [0, 1, 2] # Shoulder (XM540), Elbow (XM430), Wrist (XM430)
BAUDRATE = 1000000
DEVICENAME = 'COM5'
PROTOCOL_VERSION = 2.0

ADDR_OPERATING_MODE = 11        # Operating Mode Address
EXT_POSITION_CONTROL_MODE = 4   # Value for Extended Position Control

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

# Link Lengths (mm) - UPDATE TO YOUR PHYSICAL ARM
L1 = 159.50  
L2 = 150.50  
L3 = 110.930  
L_TUPLE = (L1, L2, L3)

# --- Relative Safety Limits ---
# Calculated from your manual run: (Limit_Tick - Zero_Tick)
# This prevents you from having to manually calibrate limits every time.
REL_LIMIT_1 = [-1575, -1456, -1340] # Corresponding to your previous "MAX" inputs
REL_LIMIT_2 = [1468, 1256, 1396]    # Corresponding to your previous "MIN" inputs

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
set_torque(False) # Torque MUST be disabled to change operating mode

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
    # Calculate absolute limits based on the current zero position
    limit_a = zero_ticks[i] + REL_LIMIT_1[i]
    limit_b = zero_ticks[i] + REL_LIMIT_2[i]
    
    max_ticks.append(limit_a)
    min_ticks.append(limit_b)
    print(f"Servo {dxl_id} limits set to: {limit_a} and {limit_b}")

time.sleep(2)
set_torque(True)
print("\nTorque ENABLED. Arm is locked and ready.")

# --- 2. Kinematics (Damped Least Squares) ---
def get_fk_and_jacobian(q, L):
    q1, q2, q3 = q
    L1, L2, L3 = L

    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    s123, c123 = np.sin(q1 + q2 + q3), np.cos(q1 + q2 + q3)

    x = L1*c1 + L2*c12 + L3*c123
    y = L1*s1 + L2*s12 + L3*s123

    J = np.array([
        [-L1*s1 - L2*s12 - L3*s123, -L2*s12 - L3*s123, -L3*s123],
        [ L1*c1 + L2*c12 + L3*c123,  L2*c12 + L3*c123,  L3*c123]
    ])
    return np.array([x, y]), J

def ik_dls(target_pos, q_init, L, max_iterations=100, tolerance=1.0, damping=0.1):
    q = np.array(q_init, dtype=float)
    target = np.array(target_pos, dtype=float)

    for _ in range(max_iterations):
        current_pos, J = get_fk_and_jacobian(q, L)
        error = target - current_pos

        if np.linalg.norm(error) < tolerance:
            return q, True

        J_T = J.T
        lambda_I = (damping ** 2) * np.eye(2)
        J_inv_dls = J_T @ np.linalg.inv(J @ J_T + lambda_I)

        q += J_inv_dls @ error

    return q, False

# --- 3. Execution Phase ---
def radians_to_ticks(radians, zero_offset, invert=False):
    direction = -1 if invert else 1
    ticks_offset = int(direction * radians * (4096 / (2 * math.pi)))
    return zero_offset + ticks_offset

def go_to_xy_3dof(target_x, target_y, current_angles):
    q_target, success = ik_dls([target_x, target_y], current_angles, L_TUPLE)
    
    if not success:
        print(f"Warning: Target ({target_x}, {target_y}) might be out of reach or near singularity.")

    for i, dxl_id in enumerate(DXL_IDs):
        # Invert rotation direction ONLY for Joint index 2 (Wrist)
        should_invert = (i == 2)
        goal_tick = radians_to_ticks(q_target[i], zero_ticks[i], invert=should_invert)
        
        # Enforce independent limits based on relative calculation
        safe_max = max(max_ticks[i], min_ticks[i])
        safe_min = min(max_ticks[i], min_ticks[i])
        clamped_tick = max(min(goal_tick, safe_max), safe_min)

        if goal_tick != clamped_tick:
             print(f"Safety constraint triggered on Joint {dxl_id}.")

        write_signed_position(dxl_id, clamped_tick)
    
    print(f"Moved to X:{target_x}, Y:{target_y} | Angles (rad): {np.round(q_target, 3)}")
    return q_target

# --- Interactive Command Loop ---
current_joint_angles = [0.0, 0.0, 0.0]

while True:
    try:
        cmd = input("\nCommand ('z'=zero, 'g'=go to XY, 'q'=quit): ").strip().lower()
        
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
                tx = float(input("Enter target X (mm): "))
                ty = float(input("Enter target Y (mm): "))
                current_joint_angles = go_to_xy_3dof(tx, ty, current_joint_angles)
            except ValueError:
                print("Invalid input. Please enter numeric values.")

    except (KeyboardInterrupt, Exception) as e:
        for i, dxl_id in enumerate(DXL_IDs):
            write_signed_position(dxl_id, zero_ticks[i])
        current_joint_angles = [0.0, 0.0, 0.0]
        print("Returned to ZERO position.")
        time.sleep(5)
        set_torque(False)
        print(f"\nTorque disabled. Exiting due to {e}.")
        break
        