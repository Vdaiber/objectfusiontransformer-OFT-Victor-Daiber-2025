import torch
from oft.transformer.training.losses.giou_losses import LossGIoUBEV
from oft.transformer.utils.normalization_utils import get_global_normalizer, normalize_coordinates, normalize_dimensions

# Dummy Config
cfg = {"loss": {"iou": {"use_exact_iou": False}}}

# Initialisiere Loss
giou_loss_fn = LossGIoUBEV(cfg)

# Hole Normalizer und Point-Cloud-Range
normalizer = get_global_normalizer()
point_cloud_range = torch.tensor(normalizer.stats['metadata']['point_cloud_range'], dtype=torch.float64)

# Erzeuge zwei Boxen im Meterraum (Ego-Koordinatensystem)
# Box 1: Zentrum (0,0,0), Größe (2,4,1.5), yaw=0
# Box 2: Zentrum (0.5,0,0), Größe (2,4,1.5), yaw=0 (überlappen)
box1_center = torch.tensor([0.0, 0.0, 0.0])
box2_center = torch.tensor([0.5, 0.0, 0.0])
box_size = torch.tensor([2.0, 4.0, 1.5])
yaw = 0.0
sin_yaw = torch.sin(torch.tensor(yaw))
cos_yaw = torch.cos(torch.tensor(yaw))

# Normalisiere Zentren und Dimensionen
box1_center_norm = normalize_coordinates(box1_center.unsqueeze(0), point_cloud_range).squeeze(0)
box2_center_norm = normalize_coordinates(box2_center.unsqueeze(0), point_cloud_range).squeeze(0)
box_size_norm = normalize_dimensions(box_size.unsqueeze(0)).squeeze(0)

# Baue normalisierte Boxen (wie vom Decoder)
# [x_norm, y_norm, z_norm, w_norm, l_norm, h_norm, sin(yaw), cos(yaw)]
box1_norm = torch.cat([box1_center_norm, box_size_norm, sin_yaw.unsqueeze(0), cos_yaw.unsqueeze(0)])
box2_norm = torch.cat([box2_center_norm, box_size_norm, sin_yaw.unsqueeze(0), cos_yaw.unsqueeze(0)])

# Simuliere Batch mit einer Box pro Batch
outputs = {"pred_boxes_normalized": box1_norm.unsqueeze(0).unsqueeze(0)}  # [B=1, N=1, 8]
targets = {"gt_boxes_b_physical": torch.cat([box2_center, box_size, torch.tensor([yaw]), torch.zeros(2)]).unsqueeze(0).unsqueeze(0)}  # [B=1, N=1, 9]

# Indices für Matching (eine Box, ein Match)
indices = [(torch.tensor([0]), torch.tensor([0]))]
num_boxes = torch.tensor(1.0)

# Loss berechnen
loss = giou_loss_fn(outputs, targets, indices, num_boxes)
print(f"GIoU-Loss (überlappende Boxen): {loss.item():.4f}")

# Teste mit weit auseinanderliegenden Boxen
box3_center = torch.tensor([20.0, 0.0, 0.0])
box3_center_norm = normalize_coordinates(box3_center.unsqueeze(0), point_cloud_range).squeeze(0)
box3_norm = torch.cat([box3_center_norm, box_size_norm, sin_yaw.unsqueeze(0), cos_yaw.unsqueeze(0)])
outputs_far = {"pred_boxes_normalized": box3_norm.unsqueeze(0).unsqueeze(0)}
loss_far = giou_loss_fn(outputs_far, targets, indices, num_boxes)
print(f"GIoU-Loss (weit entfernte Boxen): {loss_far.item():.4f}") 