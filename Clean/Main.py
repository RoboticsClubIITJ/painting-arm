import tkinter as tk
from GUI import Planar3DOFSimApp


def main():
    root = tk.Tk()
    app = Planar3DOFSimApp(root)
    # NAYA: window band karte waqt motors safely torque-off ho, isliye
    # default destroy ki jagah app.on_close use kar rahe hai.
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
