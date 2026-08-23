import cv2
import numpy as np
from pathlib import Path

WINDOW_NAME = 'XDoG Tuning'
PREVIEW_NAME = 'Sketch Preview'
updating_trackbars = False

def xdog_filter(image, sigma=1.4, k_sigma=1.6, epsilon=0.01, phi=10, gamma=0.98):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    gray_f = gray.astype(np.float32) / 255.0
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl_img = clahe.apply(gray)
    
    cl_img_f = cl_img.astype(np.float32) / 255.0

    # processed_flt = cv2.bilateralFilter(cl_img_f, d=9, sigmaColor=75, sigmaSpace=75)

    g1 = cv2.GaussianBlur(cl_img_f, (0, 0), sigma)
    g2 = cv2.GaussianBlur(cl_img_f, (0, 0), sigma * k_sigma)
    
    dog = g1 - (gamma * g2)
    
    out = np.ones_like(dog)
    idx = dog < epsilon
    out[idx] = 1.0 + np.tanh(phi * (dog[idx] - epsilon))
    
    out = (out * 255).clip(0, 255).astype(np.uint8)
    return out

def automatic_parameters(image):
    """Estimate stable starting values from image scale and local noise."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 3)
    noise = float(np.mean(cv2.absdiff(gray, median))) / 255.0
    height, width = gray.shape
    scale = max(height, width) / 1000.0
 
    sigma = np.clip(1.1 * scale + 8.0 * noise, 1.1, 3.5)
    k_sigma = np.clip(4.3 + noise * 10.0, 2.5, 5.5)
    epsilon = np.clip(0.008 + noise * 0.20, 0.005, 0.08)
    phi = int(np.clip(30 + noise * 120, 20, 60))
    gamma = np.clip(0.96 - noise * 0.25, 0.88, 0.98)
    return sigma, k_sigma, epsilon, phi, gamma

def remove_small_components(binary_image):
    """Remove isolated black regions that are too small for the robot to draw."""
    foreground = np.uint8(binary_image == 0)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    minimum_area = int(np.clip(binary_image.size * 0.00001, 8, 500))
    cleaned = np.full_like(binary_image, 255)
    for component in range(1, component_count):
        if stats[component, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == component] = 0
    return cleaned, minimum_area

def set_automatic_parameters():
    global updating_trackbars
    values = automatic_parameters(img)
    positions = {
        'Sigma x10': round(values[0] * 10),
        'k-Sigma x10': round(values[1] * 10),
        'Epsilon x1000': round(values[2] * 1000),
        'Phi': values[3],
        'Gamma x100': round(values[4] * 100),
    }
    updating_trackbars = True
    for name, position in positions.items():
        cv2.setTrackbarPos(name, WINDOW_NAME, int(position))
    updating_trackbars = False
    update_image()

def update_image(*args):
    if updating_trackbars:
        return

    # Read current trackbar positions
    s_val = cv2.getTrackbarPos('Sigma x10', 'XDoG Tuning') / 10.0
    k_val = cv2.getTrackbarPos('k-Sigma x10', 'XDoG Tuning') / 10.0
    e_val = cv2.getTrackbarPos('Epsilon x1000', 'XDoG Tuning') / 1000.0
    p_val = cv2.getTrackbarPos('Phi', 'XDoG Tuning')
    g_val = cv2.getTrackbarPos('Gamma x100', 'XDoG Tuning') / 100.0
    
    s_val = max(0.1, s_val)
    k_val = max(1.1, k_val)

    smoothed_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    
    xdog_output = xdog_filter(smoothed_img, sigma=s_val, k_sigma=k_val, epsilon=e_val, phi=p_val, gamma=g_val)
    
    otsu_val, binary_sketch = cv2.threshold(
        xdog_output, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary_sketch, minimum_area = remove_small_components(binary_sketch)

    
    info_width = 300
    info_panel = np.full((binary_sketch.shape[0], info_width, 3), 32, dtype=np.uint8)
    info = [
        'XDoG sketch parameters',
        f'Sigma: {s_val:.2f}',
        f'k-Sigma: {k_val:.2f}',
        f'Epsilon: {e_val:.3f}',
        f'Phi: {p_val}',
        f'Gamma: {g_val:.2f}',
        f'Otsu threshold: {otsu_val:.1f}',
        f'Min component: {minimum_area} px',
        '',
        'Press A: auto tune',
        'Press R: reset defaults',
        'Press Q or Esc: quit',
    ]
    for line_number, text in enumerate(info):
        cv2.putText(info_panel, text, (16, 35 + line_number * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1,
                    cv2.LINE_AA)

    preview = cv2.cvtColor(binary_sketch, cv2.COLOR_GRAY2BGR)
    cv2.imshow(PREVIEW_NAME, np.hstack((info_panel, preview)))

    if not updating_trackbars:
        print(f'Optimal threshold chosen by Otsu: {otsu_val:.1f}')

def main():
    global img
    image_path = Path(__file__).with_name('IMG_20260428_004043.jpg')
    img = cv2.imread(str(image_path))

    if img is None:
        print(f'Error: Could not open image: {image_path}')
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(PREVIEW_NAME, cv2.WINDOW_NORMAL)

    cv2.createTrackbar('Sigma x10', WINDOW_NAME, 14, 50, update_image)
    cv2.createTrackbar('k-Sigma x10', WINDOW_NAME, 16, 50, update_image)
    cv2.createTrackbar('Epsilon x1000', WINDOW_NAME, 10, 100, update_image)
    cv2.createTrackbar('Phi', WINDOW_NAME, 20, 100, update_image)
    cv2.createTrackbar('Gamma x100', WINDOW_NAME, 98, 100, update_image)

    set_automatic_parameters()
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('a'), ord('A')):
            set_automatic_parameters()
        elif key in (ord('r'), ord('R')):
            for name, position in [('Sigma x10', 14), ('k-Sigma x10', 16),
                                   ('Epsilon x1000', 10), ('Phi', 20),
                                   ('Gamma x100', 98)]:
                cv2.setTrackbarPos(name, WINDOW_NAME, position)
            update_image()
        elif key in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
