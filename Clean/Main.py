import tkinter as tk
from GUI import Planar3DOFSimApp

def main():
    root = tk.Tk()
    app = Planar3DOFSimApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()