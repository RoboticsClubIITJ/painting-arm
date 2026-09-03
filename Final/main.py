import tkinter as tk
from dynamixel_controller import DynamixelArm
from pen_controller import NanoPenController
from GUI import Planar3DOFSimApp

def main():
    print("="*40)
    print("INITIALIZING HARDWARE")
    print("="*40)
    
    # 1. Initialize Dynamixel Arm (Triggers Interactive Calibration)
    try:
        arm = DynamixelArm()
    except Exception as e:
        print(f"\n[WARNING] Dynamixel connection failed or bypassed. Running in SIM-ONLY mode for Arm.\nReason: {e}")
        arm = None
        
    # 2. Initialize Nano Pen Controller
    print("\nConnecting to Nano Pen Controller...")
    pen = NanoPenController()
    
    # 3. Launch GUI
    print("\nStarting GUI...")
    root = tk.Tk()
    app = Planar3DOFSimApp(root, arm, pen)
    
    # Graceful shutdown handler
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    
    # Start the event loop
    root.mainloop()

if __name__ == '__main__':
    main()
