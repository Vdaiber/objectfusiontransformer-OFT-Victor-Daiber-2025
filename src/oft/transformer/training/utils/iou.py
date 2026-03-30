"""
IoU calculation utilities for 3D bounding boxes with rotation support.

This module provides exact, rotation-aware IoU and GIoU calculations for 3D bounding boxes
in Bird's Eye View (BEV). The implementation uses polygon intersection algorithms to
accurately compute overlaps between rotated rectangles.

The module is optimized for performance by focusing on 2D BEV calculations rather than
full 3D IoU, which is sufficient for most object detection and tracking applications
while being significantly faster.

Key Functions:
- get_corners_2d_bev: Convert 3D boxes to 2D BEV corner points (FAST)
- cal_iou_bev: Fast 2D BEV IoU calculation for rotated boxes
- generalized_box_iou_bev: GIoU calculation using fast 2D BEV IoU

The implementation uses the Shoelace formula for polygon area calculation and
polygon clipping algorithms for intersection computation, ensuring numerical
stability and accuracy for training and evaluation.

Author: Object Fusion Transformer Team
Year: 2025
"""

import torch


def get_corners_2d_bev(boxes_3d: torch.Tensor) -> torch.Tensor:
    """
    Convert 3D bounding boxes to their 4 corner points in 2D BEV coordinates.
    
    This function transforms 3D bounding boxes defined by center coordinates,
    dimensions, and yaw angle into their 4 corner points in Bird's Eye View.
    Only considers x, y coordinates and yaw rotation for efficiency.
    
    The corner points are ordered as: front-left, front-right, back-right, back-left
    relative to the box's local coordinate system, then transformed to world coordinates.
    
    Args:
        boxes_3d: 3D bounding boxes with shape (N, 7) containing
                  [x_center, y_center, z_center, length, width, height, yaw]
    
    Returns:
        Corner points with shape (N, 4, 2) in 2D BEV coordinates.
        
    Note:
        This function is optimized for speed by only considering 2D coordinates
        and rotation, which is sufficient for BEV IoU calculations.
    """
    # Extract dimensions and center coordinates
    centers = boxes_3d[:, :2]  # (N, 2) - only x, y
    dims = boxes_3d[:, 3:5]    # (N, 2) - only length, width
    yaw = boxes_3d[:, 6]       # (N,) - yaw angle
    
    # Generate corner coordinates in local box frame (2D only)
    # Corner ordering: front-left, front-right, back-right, back-left
    l, w = dims[:, 0], dims[:, 1]
    x_corners = l.view(-1, 1) / 2 * torch.tensor([-1, 1, 1, -1], device=boxes_3d.device, dtype=boxes_3d.dtype)
    y_corners = w.view(-1, 1) / 2 * torch.tensor([-1, -1, 1, 1], device=boxes_3d.device, dtype=boxes_3d.dtype)
    corners_local = torch.stack((x_corners, y_corners), dim=1)  # (N, 2, 4)

    # Apply 2D rotation around z-axis (yaw angle)
    c, s = torch.cos(yaw), torch.sin(yaw)
    
    # 2D rotation matrix for xy-plane rotation
    rot_mat = torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s, c], dim=-1)
    ], dim=-2)  # (N, 2, 2)

    # Apply rotation to corners
    rotated_corners = torch.bmm(rot_mat, corners_local)  # (N, 2, 4)
    
    # Translate to world coordinates
    corners_world = rotated_corners.transpose(1, 2) + centers.unsqueeze(1)  # (N, 4, 2)
    return corners_world


def cal_iou_bev(boxes_a: torch.Tensor, boxes_b: torch.Tensor):
    """
    Fast 2D BEV IoU calculation for rotated boxes using polygon intersection.
    
    This function computes exact IoU for rotated boxes in Bird's Eye View using
    polygon clipping, providing accurate overlap measurements for 3D object detection.
    Much faster than 3D version as it only considers 2D coordinates.
    
    The function computes pairwise IoU between all boxes in boxes_a and boxes_b,
    returning a full NxM matrix suitable for Hungarian matching algorithms.
    
    Args:
        boxes_a: First set of 3D boxes with shape (N, 7).
        boxes_b: Second set of 3D boxes with shape (M, 7).
    
    Returns:
        Tuple of (IoU matrix, union area matrix) with shape (N, M).
        - IoU matrix: Contains pairwise IoU values between all box pairs.
        - Union area matrix: Contains union areas for all box pairs.
        
    Note:
        The function uses polygon clipping for accurate intersection calculation
        and the Shoelace formula for area computation.
    """
    corners_a = get_corners_2d_bev(boxes_a)  # (N, 4, 2)
    corners_b = get_corners_2d_bev(boxes_b)  # (M, 4, 2)
    iou_bev = torch.zeros((boxes_a.shape[0], boxes_b.shape[0]), device=boxes_a.device)
    union_area = torch.zeros((boxes_a.shape[0], boxes_b.shape[0]), device=boxes_a.device)
    
    for i in range(boxes_a.shape[0]):
        for j in range(boxes_b.shape[0]):
            # Calculate intersection area using polygon clipping
            intersection_area = _polygon_clip(corners_a[i], corners_b[j])
            
            # Calculate individual areas
            area_a = _polygon_area(corners_a[i])
            area_b = _polygon_area(corners_b[j])
            
            # Calculate union area
            union = area_a + area_b - intersection_area
            
            # Calculate IoU
            if union > 0:
                iou_bev[i, j] = intersection_area / union
            else:
                iou_bev[i, j] = 0.0
                
            union_area[i, j] = union
    
    return iou_bev, union_area


def generalized_box_iou_bev(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Calculate generalized IoU (GIoU) for two sets of boxes in BEV.
    
    GIoU extends standard IoU by considering the minimum enclosing rectangle
    of two boxes. It provides a more robust similarity measure that penalizes
    boxes that are far apart, even when their IoU is zero.
    
    The function computes a full NxM matrix for Hungarian matching algorithms.
    GIoU is particularly useful for training as it provides meaningful gradients
    even when boxes have no overlap.
    
    Args:
        boxes_a: First set of 3D boxes with shape (N, 7).
        boxes_b: Second set of 3D boxes with shape (M, 7).
    
    Returns:
        GIoU matrix with shape (N, M) containing pairwise GIoU values.
        Values range from -1 to 1, where 1 indicates perfect overlap.
        
    Note:
        GIoU = IoU - (enclosing_area - union_area) / enclosing_area
        This provides better gradients for training compared to standard IoU.
    """
    # Use fast 2D BEV IoU calculation for better performance
    iou, union_area = cal_iou_bev(boxes_a, boxes_b)

    # Get corner points for both sets of boxes (2D BEV only)
    corners_a = get_corners_2d_bev(boxes_a)  # (N, 4, 2)
    corners_b = get_corners_2d_bev(boxes_b)  # (M, 4, 2)

    # Expand dimensions for broadcasting to compute all pairs
    corners_a_expanded = corners_a.unsqueeze(1)  # (N, 1, 4, 2)
    corners_b_expanded = corners_b.unsqueeze(0)  # (1, M, 4, 2)
    
    # Combine corners from all pairs to find minimum enclosing rectangle
    all_corners = torch.cat([corners_a_expanded.expand(-1, boxes_b.shape[0], -1, -1),
                             corners_b_expanded.expand(boxes_a.shape[0], -1, -1, -1)], dim=2)  # (N, M, 8, 2)

    # Compute minimum enclosing rectangle coordinates
    c_x1 = all_corners[..., 0].min(dim=2)[0]
    c_y1 = all_corners[..., 1].min(dim=2)[0]
    c_x2 = all_corners[..., 0].max(dim=2)[0]
    c_y2 = all_corners[..., 1].max(dim=2)[0]

    # Calculate minimum enclosing rectangle area
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    
    # Compute GIoU: IoU - (enclosing_area - union_area) / enclosing_area
    giou = iou - (c_area - union_area) / (c_area + 1e-8)
    
    return giou


# ==========================================================================================
# HELPER FUNCTIONS (Used by both 2D and 3D implementations)
# ==========================================================================================

def _polygon_area(polygon: torch.Tensor):
    """
    Calculate polygon area using the Shoelace formula.
    
    The Shoelace formula (also known as the surveyor's formula) computes the
    area of a simple polygon given its vertices in order. This implementation
    is optimized for 2D polygons with torch tensors.
    
    Args:
        polygon: Polygon vertices with shape (N, 2) where N >= 3.
    
    Returns:
        Polygon area as scalar tensor.
        
    Note:
        The polygon vertices must be ordered (clockwise or counterclockwise).
        For degenerate cases with fewer than 3 vertices, returns 0.
    """
    # Polygon must have at least 3 vertices
    if polygon.shape[0] < 3:
        return torch.tensor(0.0, device=polygon.device)
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * torch.abs(torch.dot(x, torch.roll(y, 1)) - torch.dot(y, torch.roll(x, 1)))


def _get_intersection(p1, p2, edge_start, edge_end):
    """
    Calculate intersection point of two line segments.
    
    This function computes the intersection point of two line segments using
    parametric line equations. It handles edge cases such as parallel lines
    and segments that don't intersect.
    
    Args:
        p1, p2: Endpoints of first line segment.
        edge_start, edge_end: Endpoints of second line segment.
    
    Returns:
        Intersection point as tensor, or None if lines are parallel or don't intersect.
        
    Note:
        The function only returns intersection points that lie on both line segments.
        Parallel lines or non-intersecting segments return None.
    """
    p1_p2 = p2 - p1
    p3_p4 = edge_end - edge_start

    denom = (p1_p2[0] * p3_p4[1] - p1_p2[1] * p3_p4[0])
    if torch.abs(denom) < 1e-8:
        return None  # Lines are parallel

    t_num = (edge_start[0] - p1[0]) * p3_p4[1] - (edge_start[1] - p1[1]) * p3_p4[0]
    t = t_num / denom

    if not (0.0 <= t <= 1.0):  # Intersection point not on segment (p1, p2)
        return None

    return p1 + t * p1_p2


def _polygon_clip(subject_polygon, clip_polygon):
    """
    Implement Sutherland-Hodgman polygon clipping algorithm.
    
    This algorithm clips a subject polygon against a clip polygon,
    returning the intersection area.
    
    Args:
        subject_polygon: Polygon to be clipped
        clip_polygon: Clipping polygon
    
    Returns:
        Area of clipped polygon
    """
    clipped_corners = subject_polygon
    num_clip_edges = clip_polygon.shape[0]

    for i in range(num_clip_edges):
        if isinstance(clipped_corners, torch.Tensor) and clipped_corners.shape[0] == 0:
            break
        elif isinstance(clipped_corners, list) and len(clipped_corners) == 0:
            break
            
        edge_start = clip_polygon[i]
        edge_end = clip_polygon[(i + 1) % num_clip_edges]

        input_list = clipped_corners
        clipped_corners = []
        
        if isinstance(input_list, torch.Tensor) and input_list.shape[0] == 0:
            continue
        elif isinstance(input_list, list) and len(input_list) == 0:
            continue
            
        S = input_list[-1]
        edge_vec = edge_end - edge_start
        
        for E in input_list:
            s_to_edge_start = S - edge_start
            e_to_edge_start = E - edge_start

            s_inside = edge_vec[0] * s_to_edge_start[1] - edge_vec[1] * s_to_edge_start[0] <= 0
            e_inside = edge_vec[0] * e_to_edge_start[1] - edge_vec[1] * e_to_edge_start[0] <= 0

            if e_inside:
                if not s_inside:
                    intersection = _get_intersection(S, E, edge_start, edge_end)
                    if intersection is not None:
                        clipped_corners.append(intersection)
                clipped_corners.append(E)
            elif s_inside:
                intersection = _get_intersection(S, E, edge_start, edge_end)
                if intersection is not None:
                    clipped_corners.append(intersection)
            
            S = E

    if len(clipped_corners) == 0:
        return torch.tensor(0.0, device=subject_polygon.device)
    
    clipped_corners = torch.stack(clipped_corners)
    return _polygon_area(clipped_corners) 