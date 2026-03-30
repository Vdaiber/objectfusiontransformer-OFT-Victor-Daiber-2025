
def _filter_boxes_with_devkit_logic(boxes: List[DevkitBox], K: np.ndarray, img_shape) -> List[DevkitBox]:
    """Filter boxes using exact DevKit visibility logic to match GT box count."""
    from truckscenes.utils.geometry_utils import box_in_image, BoxVisibility
    
    h, w = img_shape[:2]
    imsize = (w, h)  # DevKit expects (width, height)
    
    visible_boxes = []
    for box in boxes:
        # Use DevKit's exact box_in_image function
        if box_in_image(box, K, imsize, vis_level=BoxVisibility.ANY):
            visible_boxes.append(box)
    
    return visible_boxes

# CRITICAL FIX in epoch_visualization_workflow.py:
# Line ~171: Remove .inverse from camera rotation
# OLD (BUGGY):
# camera_box.rotate(Quaternion(cs_record['rotation']).inverse)
# NEW (CORRECT):
# camera_box.rotate(Quaternion(cs_record['rotation']))

# Line ~179: Replace custom FOV filter with DevKit logic
# OLD:
# fused_boxes = _filter_boxes_in_camera_fov(fused_boxes, K, image.shape)
# NEW:
# fused_boxes = _filter_boxes_with_devkit_logic(fused_boxes, K, image.shape)
