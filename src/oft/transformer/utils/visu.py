# src/oft/transformer/utils/visu.py
"""
Visualization utilities for object detection in autonomous vehicle context.

This module provides reusable functions and constants for visualizing object
detections, with the main function being drawing 3D bounding boxes on 2D camera images.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from pyquaternion import Quaternion

# Import necessary classes from the devkit
from truckscenes.utils.data_classes import Box as DevkitBox
from truckscenes.utils.geometry_utils import view_points, transform_matrix

# Color definitions for different object classes and inputs
CLASS_COLORS_VIS_RGB: dict[str, tuple[int, int, int]] = {
    "car": (255, 158, 0), "truck": (255, 99, 71), "bus": (255, 69, 0),
    "trailer": (255, 140, 0), "other_vehicle": (233, 150, 70),
    "pedestrian": (0, 0, 230), "motorcycle": (255, 61, 99), "bicycle": (220, 20, 60),
    "traffic_cone": (47, 79, 79), "barrier": (112, 128, 144),
    "animal": (70, 130, 180), "traffic_sign": (222, 184, 135),
    '__GT__': (50, 255, 50),      # Light green for ground truth
    '__PRED__': (255, 0, 255),    # Magenta for final prediction (fallback)
}

VIRTUAL_SENSOR_INPUT_COLORS: dict[str, tuple[int, int, int, int]] = {
    "virtual_camera": (255, 100, 100, 150), # Transparent red
    "virtual_lidar": (100, 100, 255, 150),  # Transparent blue
    "virtual_radar": (255, 255, 100, 150),  # Transparent yellow
}


def get_camera_intrinsic(calib: dict) -> np.ndarray:
    """Extract camera intrinsic matrix from calibration data.
    
    Args:
        calib: Calibration dictionary containing camera intrinsic data
        
    Returns:
        3x3 camera intrinsic matrix
        
    Raises:
        ValueError: If camera intrinsic data is not found
    """
    intrinsic_data = calib.get("camera_intrinsic")
    if intrinsic_data is None:
        raise ValueError("Camera intrinsic not found in calibration data.")
    K = np.array(intrinsic_data, dtype=np.float64).reshape(3, 3)
    return K


def get_sensor_extrinsic(ego_pose: dict, calib: dict) -> np.ndarray:
    """Compute sensor extrinsic transformation matrix.
    
    Args:
        ego_pose: Ego pose dictionary containing translation and rotation
        calib: Calibration dictionary containing sensor translation and rotation
        
    Returns:
        4x4 transformation matrix from world to sensor coordinates
    """
    q_e = Quaternion(ego_pose["rotation"])
    T_world_from_ego = transform_matrix(ego_pose["translation"], q_e, inverse=False)
    T_ego_from_world = np.linalg.inv(T_world_from_ego)

    q_s = Quaternion(calib["rotation"])
    T_sensor_from_ego = transform_matrix(calib["translation"], q_s, inverse=True) 

    H_world_to_sensor = T_sensor_from_ego @ T_ego_from_world
    return H_world_to_sensor


def draw_boxes_on_image(
    image: np.ndarray,
    boxes: List[DevkitBox],
    camera_k_matrix: np.ndarray,
    world_to_sensor_transform: np.ndarray,
    color_override_rgb: Optional[Tuple[int, int, int]] = None,
    line_thickness: int = 2,
    z_threshold: float = 0.1,
    is_gt: bool = False,
) -> np.ndarray:
    """
    Projiziert und zeichnet eine Liste von 3D-Bounding-Boxes auf ein 2D-Bild.

    Args:
        image: Das OpenCV-Bild (BGR), auf das gezeichnet werden soll.
        boxes: Eine Liste von DevkitBox-Objekten in Weltkoordinaten.
        camera_k_matrix: Die intrinsische Kameramatrix (3x3).
        world_to_sensor_transform: Die extrinsische Transformationsmatrix (4x4),
                                   die von Welt- in Kamerasensor-Koordinaten transformiert.
        color_override_rgb: Falls gesetzt, wird diese RGB-Farbe für alle Boxen verwendet.
        line_thickness: Dicke der gezeichneten Linien.
        z_threshold: Mindestabstand vor der Kamera, damit eine Box gezeichnet wird.
        is_gt: Wenn True, werden die Boxen als Ground-Truth (z.B. gestrichelt/transparent) gezeichnet.

    Returns:
        Das Bild mit den eingezeichneten Boxen.
    """
    img_out = image.copy()
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Untere Fläche
        (4, 5), (5, 6), (6, 7), (7, 4),  # Obere Fläche
        (0, 4), (1, 5), (2, 6), (3, 7)   # Verbindungen
    ]

    for box in boxes:
        # Transformiere Box-Ecken von Welt- in Sensor-Koordinaten
        box_corners_world = box.corners()
        box_corners_world_h = np.vstack((box_corners_world, np.ones((1, 8))))
        box_corners_sensor_h = world_to_sensor_transform @ box_corners_world_h
        box_corners_sensor = box_corners_sensor_h[:3, :]

        # Überspringe Boxen, die komplett hinter der Kamera liegen
        if np.all(box_corners_sensor[2, :] <= z_threshold):
            print(f"[draw_boxes_on_image] Skipping box (all corners behind camera): center={getattr(box, 'center', 'N/A')}, name={getattr(box, 'name', 'N/A')}")
            continue

        # Projiziere 3D-Punkte auf die 2D-Bildebene
        image_points_raw = view_points(box_corners_sensor, camera_k_matrix, normalize=True)
        image_points = image_points_raw[:2, :].astype(int)

        # Bestimme die Farbe
        if is_gt:
            rgb_color = CLASS_COLORS_VIS_RGB['__GT__']
        elif color_override_rgb:
            rgb_color = color_override_rgb
        else:
            rgb_color = CLASS_COLORS_VIS_RGB.get(box.name, CLASS_COLORS_VIS_RGB['__PRED__'])
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0]) # Konvertiere zu BGR für OpenCV

        if is_gt:
            print(f"[draw_boxes_on_image] Drawing GT box: center={getattr(box, 'center', 'N/A')}, name={getattr(box, 'name', 'N/A')}, color={rgb_color}")
        
        # Zeichne die Kanten der Box
        for i, j in edges:
            # Zeichne eine Kante nur, wenn beide Endpunkte vor der Kamera liegen
            if box_corners_sensor[2, i] > z_threshold and box_corners_sensor[2, j] > z_threshold:
                p1 = tuple(image_points[:, i])
                p2 = tuple(image_points[:, j])
                
                if is_gt: # Spezielles Zeichnen für Ground Truth
                    # Erstelle ein Overlay für transparente Linien
                    overlay = img_out.copy()
                    cv2.line(overlay, p1, p2, bgr_color, line_thickness, cv2.LINE_AA)
                    alpha = 0.6  # 60% Sichtbarkeit
                    img_out = cv2.addWeighted(overlay, alpha, img_out, 1 - alpha, 0)
                else: # Standard-Zeichnen
                    cv2.line(img_out, p1, p2, bgr_color, line_thickness, cv2.LINE_AA)

    return img_out