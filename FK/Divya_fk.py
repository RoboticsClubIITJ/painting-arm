import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class PlanarRobotArm:
    """
    A customizable Forward Kinematics solver for a planar (2D) robot arm.
    Uses 3x3 Homogeneous Transformation Matrices to calculate joint positions.
    Enhanced version with additional analysis features.
    """
    def __init__(self, link_lengths):
        """
        Initialize the arm with a list of link lengths.
        Example for a 3-link arm: PlanarRobotArm([10.0, 10.0, 10.0])
        
        Parameters:
            link_lengths (list): List of link lengths in cm
        """
        self.link_lengths = link_lengths
        self.num_links = len(link_lengths)

    def _get_transform_matrix(self, theta_rad, link_length):
        """
        Creates a 3x3 homogeneous transformation matrix for a 2D plane.
        This combines both rotation (theta) and translation (link_length).
        
        Parameters:
            theta_rad (float): Joint angle in radians
            link_length (float): Length of the link in cm
            
        Returns:
            np.ndarray: 3x3 transformation matrix
        """
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        
        return np.array([
            [c, -s, link_length * c],
            [s,  c, link_length * s],
            [0,  0, 1              ]
        ])

    def compute_fk(self, thetas_deg):
        """
        Computes the forward kinematics given a list of joint angles in degrees.
        Returns a list of (x, y) tuples representing the position of every joint,
        ending with the final paintbrush/end-effector position.
        
        Parameters:
            thetas_deg (list): Joint angles in degrees [θ1, θ2, θ3, ...]
            
        Returns:
            list: List of (x, y) tuples for each joint position
            
        Raises:
            ValueError: If number of angles doesn't match number of links
        """
        if len(thetas_deg) != self.num_links:
            raise ValueError(f"Expected {self.num_links} angles, but got {len(thetas_deg)}.")

        # Convert degrees to radians for numpy trigonometric functions
        thetas_rad = np.radians(thetas_deg)

        # The base of the robot is always at origin (0, 0)
        joint_positions = [(0.0, 0.0)]
        
        # Start with an Identity Matrix (no translation, no rotation)
        T_accumulated = np.eye(3)

        for i in range(self.num_links):
            # 1. Get the local transformation for the current joint & link
            T_current = self._get_transform_matrix(thetas_rad[i], self.link_lengths[i])
            
            # 2. Multiply with previous transformations to get global coordinates
            # This is exactly what the slides describe: p' = H_n * H_n-1 * ... * H_1 * p
            T_accumulated = T_accumulated @ T_current
            
            # 3. Extract the X and Y translations from the resulting matrix
            x_pos = T_accumulated[0, 2]
            y_pos = T_accumulated[1, 2]
            
            joint_positions.append((x_pos, y_pos))

        return joint_positions

    def get_end_effector(self, thetas_deg):
        """
        Get only the end-effector (brush tip) position.
        
        Parameters:
            thetas_deg (list): Joint angles in degrees
            
        Returns:
            tuple: (x, y) position of end-effector
        """
        positions = self.compute_fk(thetas_deg)
        return positions[-1]

    def get_cumulative_angles(self, thetas_deg):
        """
        Get cumulative angles at each joint.
        
        Parameters:
            thetas_deg (list): Joint angles in degrees
            
        Returns:
            np.ndarray: Cumulative angles in degrees
        """
        thetas_rad = np.radians(thetas_deg)
        cumulative_angles = np.cumsum(thetas_rad)
        return np.degrees(cumulative_angles)

    def get_total_reach(self, thetas_deg):
        """
        Calculate the total reach distance from base to end-effector.
        
        Parameters:
            thetas_deg (list): Joint angles in degrees
            
        Returns:
            float: Distance from origin to end-effector
        """
        x, y = self.get_end_effector(thetas_deg)
        return np.sqrt(x**2 + y**2)

    def get_max_reach(self):
        """
        Calculate the maximum possible reach of the arm.
        
        Returns:
            float: Sum of all link lengths
        """
        return sum(self.link_lengths)


# ==========================================
# Example Usage for the Painting Arm Project
# ==========================================
if __name__ == "__main__":
    # Define a 3-link arm (e.g., shoulder, elbow, wrist) in centimeters
    robot_arm = PlanarRobotArm(link_lengths=[15.0, 12.0, 8.0])

    # Setup the Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.5)  # Make room for sliders at the bottom
    
    # Calculate initial positions (starting at 0 degrees)
    initial_angles = [0.0, 0.0, 0.0]
    positions = robot_arm.compute_fk(initial_angles)
    
    # Extract X and Y arrays for plotting
    X_coords = [pos[0] for pos in positions]
    Y_coords = [pos[1] for pos in positions]

    # Plot the arm segments, base, and brush tip
    arm_line, = ax.plot(X_coords, Y_coords, 'o-', linewidth=5, markersize=8, color='#2c3e50', label='Robot Links')
    ax.plot(0, 0, 'rs', markersize=10, label='Base (Origin)')
    brush_tip, = ax.plot(X_coords[-1], Y_coords[-1], 'g*', markersize=15, label='Brush Tip')

    # Add styling and labels
    max_reach = robot_arm.get_max_reach() + 5
    ax.set_xlim(-max_reach, max_reach)
    ax.set_ylim(-max_reach, max_reach)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title('Painting Arm - Forward Kinematics', fontweight='bold')
    ax.set_xlabel('X Coordinate (cm)')
    ax.set_ylabel('Y Coordinate (cm)')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    # --- Sliders Setup ---
    # Axes arrays: [left, bottom, width, height]
    # Angle Sliders (Bottom Group)
    ax_t1 = plt.axes([0.15, 0.25, 0.70, 0.03])
    ax_t2 = plt.axes([0.15, 0.20, 0.70, 0.03])
    ax_t3 = plt.axes([0.15, 0.15, 0.70, 0.03])

    # Length Sliders (Top Group)
    ax_l1 = plt.axes([0.15, 0.40, 0.70, 0.03])
    ax_l2 = plt.axes([0.15, 0.35, 0.70, 0.03])
    ax_l3 = plt.axes([0.15, 0.30, 0.70, 0.03])

    # Angle Sliders
    slider_t1 = Slider(ax_t1, 'Theta 1', -180.0, 180.0, valinit=initial_angles[0], valfmt='%0.1f°')
    slider_t2 = Slider(ax_t2, 'Theta 2', -180.0, 180.0, valinit=initial_angles[1], valfmt='%0.1f°')
    slider_t3 = Slider(ax_t3, 'Theta 3', -180.0, 180.0, valinit=initial_angles[2], valfmt='%0.1f°')

    # Length Sliders
    slider_l1 = Slider(ax_l1, 'Length 1 (cm)', 1.0, 30.0, valinit=robot_arm.link_lengths[0], valfmt='%0.1f cm')
    slider_l2 = Slider(ax_l2, 'Length 2 (cm)', 1.0, 30.0, valinit=robot_arm.link_lengths[1], valfmt='%0.1f cm')
    slider_l3 = Slider(ax_l3, 'Length 3 (cm)', 1.0, 30.0, valinit=robot_arm.link_lengths[2], valfmt='%0.1f cm')


    # Update Function
    def update(val):
        # 1. Grab angles from sliders
        t1 = slider_t1.val
        t2 = slider_t2.val
        t3 = slider_t3.val
        
        # 2. Grab lengths from sliders
        l1 = slider_l1.val
        l2 = slider_l2.val
        l3 = slider_l3.val
        
        # 3. Update the link lengths in the class
        robot_arm.link_lengths = [l1, l2, l3]

        # 4. Recalculate kinematics using the Class!
        new_positions = robot_arm.compute_fk([t1, t2, t3])
        
        # 5. Extract X and Y arrays
        new_X = [pos[0] for pos in new_positions]
        new_Y = [pos[1] for pos in new_positions]
        
        # 6. Redraw the arm and the brush tip
        arm_line.set_data(new_X, new_Y)
        brush_tip.set_data([new_X[-1]], [new_Y[-1]])  # Matplotlib requires a sequence/list for single points
        
        # 7. Dynamically adjust axes limits if the arm extends beyond the current view
        current_max_reach = robot_arm.get_max_reach() + 5
        ax.set_xlim(-current_max_reach, current_max_reach)
        ax.set_ylim(-current_max_reach, current_max_reach)

        fig.canvas.draw_idle()

    # Link all sliders to the update trigger
    slider_t1.on_changed(update)
    slider_t2.on_changed(update)
    slider_t3.on_changed(update)
    slider_l1.on_changed(update)
    slider_l2.on_changed(update)
    slider_l3.on_changed(update)

    print("\nOpening Interactive GUI Visualization...")
    plt.show()
