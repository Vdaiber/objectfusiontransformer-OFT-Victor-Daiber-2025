import os
import numpy as np
import cv2
from truckscenes.utils.data_classes import Box as DevkitBox
from truckscenes.utils.visualization_utils import render_box_cv2, TruckScenesExplorer, view_points
from truckscenes.utils.colormap import get_colormap
# NEW: Directly import the official category mapping function from the devkit
from truckscenes.eval.detection.utils import category_to_detection_name
import logging

def render_box_cv2_dashed(box,
                          im: np.ndarray,
                          view: np.ndarray = np.eye(3),
                          normalize: bool = False,
                          colors: tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
                          linewidth: int = 2,
                          dashed: bool = False) -> None:
    """
    Erweiterte Version von render_box_cv2 mit Unterstützung für gestrichelte Linien.
    :param dashed: Wenn True, werden gestrichelte Linien gezeichnet
    """
    corners = view_points(box.corners(), view, normalize=normalize)[:2, :]

    def draw_line_conditional(pt1, pt2, color, thickness, dashed_flag):
        """Zeichnet eine Linie - entweder gestrichelt oder durchgezogen"""
        if dashed_flag:
            # Gestrichelte Linie: Zeichne mehrere kurze Segmente
            draw_dashed_line(im, pt1, pt2, color, thickness)
        else:
            # Durchgezogene Linie
            cv2.line(im, pt1, pt2, color, thickness)

    def draw_rect_conditional(selected_corners, color, dashed_flag):
        """Zeichnet ein Rechteck - entweder gestrichelt oder durchgezogen"""
        prev = selected_corners[-1]
        for corner in selected_corners:
            draw_line_conditional(
                (int(prev[0]), int(prev[1])),
                (int(corner[0]), int(corner[1])),
                color, linewidth, dashed_flag
            )
            prev = corner

    # Draw the sides
    for i in range(4):
        draw_line_conditional(
            (int(corners.T[i][0]), int(corners.T[i][1])),
            (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
            colors[2], linewidth, dashed
        )

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles
    draw_rect_conditional(corners.T[:4], colors[0], dashed)
    draw_rect_conditional(corners.T[4:], colors[1], dashed)


def draw_dashed_line(img, pt1, pt2, color, thickness, dash_length=8, gap_length=4):
    """
    Zeichnet eine gestrichelte Linie zwischen zwei Punkten.
    """
    pt1, pt2 = tuple(map(int, pt1)), tuple(map(int, pt2))
    
    # Berechne Linienlänge und -richtung
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    distance = np.sqrt(dx*dx + dy*dy)
    
    if distance == 0:
        return
    
    # Normalisierte Richtung
    dx_norm = dx / distance
    dy_norm = dy / distance
    
    # Zeichne gestrichelte Linie
    current_pos = 0
    drawing = True
    
    while current_pos < distance:
        if drawing:
            # Zeichne Dash-Segment
            segment_end = min(current_pos + dash_length, distance)
            start_x = pt1[0] + int(current_pos * dx_norm)
            start_y = pt1[1] + int(current_pos * dy_norm)
            end_x = pt1[0] + int(segment_end * dx_norm)
            end_y = pt1[1] + int(segment_end * dy_norm)
            
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
            current_pos = segment_end
        else:
            # Überspringe Gap-Segment
            current_pos = min(current_pos + gap_length, distance)
        
        drawing = not drawing


def get_devkit_bgr_color(detection_name: str) -> tuple:
    """
    Gets a consistent BGR color for a valid detection name using the
    official truckscenes-devkit colormap.
    """
    # Map the simplified name back to a representative devkit category name
    # to look up the official color. This logic is now simplified as we
    # rely on the official mapping first.
    simple_to_category_for_color = {
        'car': 'vehicle.car',
        'truck': 'vehicle.truck',
        'bus': 'vehicle.bus.rigid',
        'trailer': 'vehicle.trailer',
        'other_vehicle': 'vehicle.other',
        'pedestrian': 'human.pedestrian.adult',
        'motorcycle': 'vehicle.motorcycle',
        'bicycle': 'vehicle.bicycle',
        'traffic_cone': 'movable_object.trafficcone',
        'barrier': 'movable_object.barrier',
        'animal': 'animal',
        'traffic_sign': 'static_object.traffic_sign'
    }
    # Use the provided detection_name to find the representative category for color lookup
    category_for_color = simple_to_category_for_color.get(detection_name, 'vehicle.car')

    # Get the official RGB color from the devkit colormap
    colormap = get_colormap()
    rgb_color = colormap.get(category_for_color, (255, 158, 0))  # Default to orange (car)

    # Convert RGB to BGR for OpenCV
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
    return bgr_color

def visualize_epoch_sample(
    image: np.ndarray,
    fused_boxes: list,
    gt_boxes: list,
    camera_k_matrix: np.ndarray,
    run_dir: str,
    epoch: int,
    trucksc=None
):
    """
    Visualisiert Fused- und GT-Boxen auf einem Bild und speichert das Ergebnis im Visualisierungs-Unterordner des aktuellen Runs.
    Fused-Boxen: Blau, GT-Boxen: Devkit-Style.
    
    WICHTIG: render_box_cv2 erwartet Boxen in Kamera-Koordinaten!
    """
    vis_dir = os.path.join(run_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    output_path = os.path.join(vis_dir, f"epoch_{epoch:04d}.jpg")
    img = image.copy()

    
    # 1. Predicted-Boxen mit gestrichelten Linien (DASHED)
    for i, box in enumerate(fused_boxes):
        try:
            # The prediction name is already a simplified detection name.
            # No conversion needed.
            detection_name = box.name
            
            # Get the official devkit color for the detection name
            bgr_color = get_devkit_bgr_color(detection_name)
            
            render_box_cv2_dashed(
                box,
                img,
                view=camera_k_matrix,
                normalize=True,
                colors=(bgr_color, bgr_color, bgr_color),
                linewidth=2,
                dashed=True  # Dashed lines for predictions
            )
        except Exception as e:
            logging.warning(f"Failed to render predicted box: {e}", exc_info=True)
            # Fallback: Blue dashed for errors
            render_box_cv2_dashed(
                box,
                img,
                view=camera_k_matrix,
                normalize=True,
                colors=((0, 0, 255), (0, 0, 255), (0, 0, 255)),  # Blue (BGR)
                linewidth=2,
                dashed=True
            )

    # 2. GT-Boxen mit durchgezogenen Linien (SOLID)
    for i, box in enumerate(gt_boxes):
        try:
            # Use the OFFICIAL devkit function to convert the full category name.
            # This will return None for categories that should be ignored in evaluation.
            detection_name = category_to_detection_name(box.name)
            
            # Only draw the box if it belongs to a valid detection category
            if detection_name is not None:
                # Get the official devkit color
                bgr_color = get_devkit_bgr_color(detection_name)
                
                render_box_cv2_dashed(
                    box, 
                    img, 
                    view=camera_k_matrix, 
                    normalize=True, 
                    colors=(bgr_color, bgr_color, bgr_color), 
                    linewidth=2,
                    dashed=False  # Solid lines for GT
                )
        except Exception as e:
            logging.warning(f"Failed to render ground truth box: {e}", exc_info=True)
            # Fallback: Green solid for errors
            render_box_cv2_dashed(
                box, 
                img, 
                view=camera_k_matrix, 
                normalize=True, 
                colors=((0, 255, 0), (0, 255, 0), (0, 255, 0)),  # Green (BGR)
                linewidth=2,
                dashed=False
            )

    cv2.imwrite(output_path, img)
    print(f"Visualisierung gespeichert: {output_path}") 