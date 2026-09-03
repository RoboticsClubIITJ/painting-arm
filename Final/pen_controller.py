import serial
import time
from Configurations import ARDUINO_COM_PORT, PEN_UP_ANGLE, PEN_DOWN_ANGLE

class NanoPenController:
    def __init__(self, port=ARDUINO_COM_PORT, baud_rate=9600):
        self.ser = None
        self.current_state = None # Keep track of state to avoid redundant serial writes
        try:
            self.ser = serial.Serial(port, baud_rate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset upon connection
            print(f"[HARDWARE] Nano Pen Controller connected on {port}.")
        except Exception as e:
            print(f"[HARDWARE] Nano Pen connection failed (sim-only mode for pen): {e}")

    def send_angle(self, angle):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(f"{(angle)}\n".encode())
            except Exception as e:
                print(f"[HARDWARE] Failed to send angle: {e}")

    def pen_up(self):
        if self.current_state != 'up':
            self.send_angle('U')
            self.current_state = 'up'

    def pen_down(self):
        if self.current_state != 'down':
            self.send_angle('D')
            self.current_state = 'down'

    def close(self):
        if self.ser:
            self.ser.close()
