import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. OPENCV WAYPOINT EXTRACTION ENGINE
# =====================================================================

def extract_waypoints_from_image(
    image_path, 
    num_waypoints=100, 
    workspace_x=(10.0, 25.0), 
    workspace_y=(-10.0, 15.0), 
    target_phi=0.0
):
    """
    Reads an image stroke, extracts ordered contour points, normalizes them, 
    and maps them directly to the robot arm's canvas workspace coordinates.
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image file: {image_path}")

    # 1. Thresholding: Convert dark stroke on light paper into a binary mask
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 2. Find Contours (Sequential pixel ordering along stroke edge)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No stroke contours found in the image!")

    # Select the primary (largest) stroke contour
    main_contour = max(contours, key=lambda c: cv2.arcLength(c, True))
    pts = main_contour.reshape(-1, 2)  # Reshape to (N, 2) array of [x_pixel, y_pixel]

    # 3. Downsample / Resample to a clean waypoint count
    total_pts = len(pts)
    if total_pts > num_waypoints:
        indices = np.linspace(0, total_pts - 1, num_waypoints, dtype=int)
        sampled_pts = pts[indices]
    else:
        sampled_pts = pts

    # 4. Map Image Pixel Space -> Robot Canvas Space
    h, w = img.shape
    x_pixels = sampled_pts[:, 0]
    y_pixels = sampled_pts[:, 1]

    # Normalize pixels to range [0, 1]
    # CRITICAL: Flip Y axis because Image Y points DOWN, but Robot Canvas Y points UP
    x_norm = x_pixels / float(w)
    y_norm = 1.0 - (y_pixels / float(h))

    # Scale normalized coordinates to physical canvas boundaries (in cm)
    x_robot = workspace_x[0] + x_norm * (workspace_x[1] - workspace_x[0])
    y_robot = workspace_y[0] + y_norm * (workspace_y[1] - workspace_y[0])

    # Package into key waypoints array [X, Y, Phi]
    waypoints = []
    for x, y in zip(x_robot, y_robot):
        waypoints.append([round(x, 3), round(y, 3), target_phi])

    return np.array(waypoints), img, thresh


# =====================================================================
# 2. SYNTHETIC IMAGE GENERATOR (FOR IMMEDIATE TESTING)
# =====================================================================

def create_dummy_stroke_image(filename="test_stroke.png"):
    """Creates a simple black stroke on a white canvas for testing."""
    img = np.ones((400, 400), dtype=np.uint8) * 255
    # Draw a curved stroke path
    pts = np.array([[80, 100], [320, 120], [100, 280], [320, 300]], np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=0, thickness=6)
    cv2.imwrite(filename, img)
    return filename


# =====================================================================
# 3. VISUALIZATION & PIPELINE TEST
# =====================================================================

if __name__ == "__main__":
    # Generate a sample image file automatically
    sample_img_path = create_dummy_stroke_image("test_stroke.png")

    # Define robot canvas physical workspace boundaries (cm)
    CANVAS_X_BOUNDS = (10.0, 25.0)  # Min X, Max X
    CANVAS_Y_BOUNDS = (-5.0, 15.0)  # Min Y, Max Y

    # Extract 50 key waypoints from image
    key_waypoints, original_img, thresh_img = extract_waypoints_from_image(
        image_path=sample_img_path,
        num_waypoints=50,
        workspace_x=CANVAS_X_BOUNDS,
        workspace_y=CANVAS_Y_BOUNDS,
        target_phi=np.radians(0.0) # Flat brush orientation
    )

    print("--- EXTRACTED ROBOT WAYPOINTS (First 5) ---")
    print("Format: [X_cm, Y_cm, Phi_rad]")
    for wp in key_waypoints[:5]:
        print(wp)

    # Plot verification
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Left: Original Image
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title("Input Image Stroke")
    ax1.axis('off')

    # Right: Extracted Waypoints mapped to Robot Canvas
    ax2.plot(key_waypoints[:, 0], key_waypoints[:, 1], 'r-o', markersize=4, label='Extracted Waypoints')
    ax2.set_xlim(CANVAS_X_BOUNDS[0] - 2, CANVAS_X_BOUNDS[1] + 2)
    ax2.set_ylim(CANVAS_Y_BOUNDS[0] - 2, CANVAS_Y_BOUNDS[1] + 2)
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle='--')
    ax2.set_title("Robot Canvas Space (cm)")
    ax2.set_xlabel("X (cm)")
    ax2.set_ylabel("Y (cm)")
    ax2.legend()

    plt.tight_layout()
    plt.show()
