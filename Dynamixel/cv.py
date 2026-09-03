"""
Image -> drawable-path pipeline using XDoG for detailed drawings/portraits.

Pipeline:
1. Preprocess: Bilateral filter -> XDoG
2. Binarize: Otsu threshold
3. Clean: Remove small connected components
4. Extract: Contours -> RDP simplification -> B-spline smoothing
5. Map: Scale/recenter into the robot's physical drawing workspace
"""

import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from Configurations import (MIN_CONTOUR_ARC_LEN,MIN_CONTOUR_POINTS,SPLINE_SMOOTHING,APPROX_POLY_EPSILON,TARGET_CANVAS_W,TARGET_CANVAS_H,CANVAS_CENTER_X,CANVAS_CENTER_Y,)


def xdog_filter(image, sigma=1.4, k_sigma=1.6, epsilon=0.01, phi=20, gamma=0.98):
    """
    Extended Difference-of-Gaussians (XDoG).

    Produces a sketch-like grayscale image emphasizing dark line
    structures and fine details.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
    cl_img = clahe.apply(gray)

    cl_img_f = cl_img.astype(np.float32) / 255.0

    g1 = cv2.GaussianBlur(cl_img_f, (0, 0), sigma)
    g2 = cv2.GaussianBlur(cl_img_f, (0, 0), sigma * k_sigma)

    dog = g1 - (gamma * g2)
    out = np.ones_like(dog)
    idx = dog < epsilon

    out[idx] = 1.0 + np.tanh(phi * (dog[idx] - epsilon))

    out = (out * 255).clip(0, 255).astype(np.uint8)
    return out

def automatic_parameters(image):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    median = cv2.medianBlur(gray, 3)
    noise = float(np.mean(cv2.absdiff(gray, median))) / 255.0
    height, width = gray.shape
    scale = max(height, width) / 1000.0

    sigma = np.clip(1.1 * scale + 8.0 * noise,1.1,3.5)
    k_sigma = np.clip(4.3 + noise * 10.0, 2.5, 5.5)
    epsilon = np.clip(0.008 + noise * 0.20,0.005,0.08)
    phi = int(np.clip(30 + noise * 120, 20, 60))
    gamma = np.clip(0.96 - noise * 0.25, 0.88, 0.98)

    return sigma, k_sigma, epsilon, phi, gamma

def remove_small_components(binary_image):
    foreground = np.uint8(binary_image == 0)
    component_count, labels, stats, _ = (cv2.connectedComponentsWithStats(foreground,connectivity=8))

    minimum_area = int(np.clip(binary_image.size * 0.00001,8,500))
    cleaned = np.full_like(binary_image,255)
    for component in range(1, component_count):
        area = stats[component,cv2.CC_STAT_AREA]
        if area >= minimum_area:
            cleaned[labels == component] = 0
    return cleaned

def preprocess_image(raw_image, sigma=None, k_sigma=None, epsilon=None, phi=None, gamma=None):
    # Ensure grayscale
    if raw_image.ndim == 3:
        gray = cv2.cvtColor(raw_image,cv2.COLOR_BGR2GRAY)
    else:
        gray = raw_image.copy()
    # Moderate bilateral filtering.
    # Kept from the tested XDoG pipeline because it suppresses
    # sensor noise while preserving facial/detail structures.
    smoothed = cv2.bilateralFilter(
        gray,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )

    # Automatically determine missing parameters
    if ( sigma is None or k_sigma is None or epsilon is None or phi is None or gamma is None):
        auto = automatic_parameters(smoothed)
        sigma = auto[0] if sigma is None else sigma
        k_sigma = auto[1] if k_sigma is None else k_sigma
        epsilon = auto[2] if epsilon is None else epsilon
        phi = auto[3] if phi is None else phi
        gamma = auto[4] if gamma is None else gamma
    xdog_output = xdog_filter(smoothed, sigma=sigma, k_sigma=k_sigma, epsilon=epsilon, phi=phi, gamma=gamma)

    # Automatic binarization
    _, binary_sketch = cv2.threshold(xdog_output, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove isolated noise
    binary_sketch = remove_small_components(binary_sketch)
    return binary_sketch

def extract_smoothed_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    paths = []
    for cnt in contours:
        cnt = cnt.squeeze()
        if cnt.ndim < 2:
            continue

        # Filter very short contours
        arc_len = cv2.arcLength(cnt, closed=False)

        if arc_len < MIN_CONTOUR_ARC_LEN:
            continue

        # Remove consecutive duplicate points
        diffs = np.diff(cnt, axis=0)
        mask = np.any(diffs != 0, axis=1)

        cnt = cnt[np.append([True], mask)]
        if len(cnt) < MIN_CONTOUR_POINTS:
            continue

        # RDP simplification
        poly_approx = cv2.approxPolyDP(cnt.astype(np.float32), APPROX_POLY_EPSILON, closed=False).squeeze()

        if (poly_approx.ndim < 2 or len(poly_approx) < 3):
            poly_approx = cnt

        # B-spline smoothing
        x = poly_approx[:, 0]
        y = poly_approx[:, 1]

        try:
            is_closed = (np.linalg.norm(poly_approx[0]- poly_approx[-1]) < 3.0)
            spline_order = min(3, len(poly_approx) - 1)
            tck, u = splprep([x, y], s=SPLINE_SMOOTHING, k=spline_order, per=is_closed)
            num_points = max(8, int(arc_len * 0.4))
            u_new = np.linspace(u.min(), u.max(), num_points)
            x_new, y_new = splev(u_new, tck)
            smooth_approx = np.vstack((x_new, y_new)).T
        except Exception:
            smooth_approx = (poly_approx.astype(float))
        # Bounding box used later for stroke ordering
        x_bound, y_bound, _, _ = cv2.boundingRect(smooth_approx.astype(np.int32))

        paths.append({'points': smooth_approx,'y': y_bound,'x': x_bound})

    return paths

def map_paths_to_workspace(paths, image_shape):
    # Preserve current ordering behavior
    paths = sorted(paths,key=lambda p: (p['y'], p['x']))

    h, w = image_shape[:2]
    scale = min(TARGET_CANVAS_W / w,TARGET_CANVAS_H / h)
    mapped_paths = []
    for p_dict in paths:
        p = p_dict['points'].astype(float)
        mapped = []
        for point in p:
            # Image coordinates -> image-centered coordinates
            cx_img = (point[0] - (w / 2.0))
            cy_img = (point[1] - (h / 2.0))
            # Image-centered -> robot workspace
            mapped.append((CANVAS_CENTER_X + (cx_img * scale), CANVAS_CENTER_Y- (cy_img * scale)))

        # Close loop when endpoints are sufficiently close
        if (len(mapped) > 2 and np.hypot(mapped[0][0] - mapped[-1][0], mapped[0][1] - mapped[-1][1]) < (3.0 * scale)):
            mapped.append(mapped[0])
        mapped_paths.append(mapped)
    return mapped_paths

def image_to_robot_paths(raw_image, sigma=None, k_sigma=None, epsilon=None, phi=None, gamma=None):
    binary_image = preprocess_image(raw_image, sigma=sigma, k_sigma=k_sigma, epsilon=epsilon, phi=phi, gamma=gamma)
    paths = extract_smoothed_contours(binary_image)
    return map_paths_to_workspace(paths, raw_image.shape)