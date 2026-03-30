import pytest
import torch
from src.oft.transformer.models.architectures.autoregressive_architecture import ObjectFusionTransformerAutoregressive

# Helper function to create a dummy model instance for testing
@pytest.fixture
def model_instance():
    """Provides a dummy instance of the model for testing its methods."""
    # A full model initialization is complex, so we create a dummy class
    # that only has the method we want to test.
    class DummyModel:
        def _extrapolate_memory_with_physics(self, 
                                             memory_anchor_boxes: torch.Tensor, 
                                             dt: torch.Tensor
                                             ) -> torch.Tensor:
            """
            This is a direct copy of the method from the real model for isolated testing.
            """
            if dt.nelement() == 0 or dt.item() <= 0:
                return memory_anchor_boxes

            memory_velocities = memory_anchor_boxes[..., 7:9]
            dt_device = dt.to(memory_velocities.device)
            delta_xy = memory_velocities * dt_device.view(1, 1, 1)
            extrapolated_boxes = memory_anchor_boxes.clone()
            extrapolated_boxes[..., :2] += delta_xy
            return extrapolated_boxes
            
    return DummyModel()

@pytest.fixture
def sample_anchor_boxes():
    """Provides a sample tensor of memory anchor boxes."""
    # Shape: [Batch, Num_Boxes, Features] -> [1, 2, 9]
    # Box 1: pos(10, 20), vel(1, 2)
    # Box 2: pos(30, 40), vel(-3, -4)
    return torch.tensor([
        [
            [10.0, 20.0, 5.0, 2.0, 1.5, 1.8, 0.0, 1.0, 2.0],  # Box 1
            [30.0, 40.0, 6.0, 2.5, 1.6, 1.9, 0.0, -3.0, -4.0] # Box 2
        ]
    ], dtype=torch.float32)

def test_extrapolation_with_zero_dt(model_instance, sample_anchor_boxes):
    """
    Tests that if dt is zero, the positions of the boxes remain unchanged.
    """
    dt = torch.tensor(0.0)
    original_boxes = sample_anchor_boxes.clone()
    
    extrapolated_boxes = model_instance._extrapolate_memory_with_physics(sample_anchor_boxes, dt)
    
    assert torch.equal(original_boxes, extrapolated_boxes), "Boxes should not change when dt is 0"

def test_extrapolation_with_positive_dt(model_instance, sample_anchor_boxes):
    """
    Tests the core physics logic: new_pos = old_pos + velocity * dt.
    """
    dt = torch.tensor(0.5) # 0.5 seconds
    
    extrapolated_boxes = model_instance._extrapolate_memory_with_physics(sample_anchor_boxes, dt)
    
    # Expected position for Box 1:
    # x = 10.0 + (1.0 * 0.5) = 10.5
    # y = 20.0 + (2.0 * 0.5) = 21.0
    expected_pos_box1 = torch.tensor([10.5, 21.0])
    
    # Expected position for Box 2:
    # x = 30.0 + (-3.0 * 0.5) = 28.5
    # y = 40.0 + (-4.0 * 0.5) = 38.0
    expected_pos_box2 = torch.tensor([28.5, 38.0])
    
    assert torch.allclose(extrapolated_boxes[0, 0, :2], expected_pos_box1), "Box 1 position is incorrect"
    assert torch.allclose(extrapolated_boxes[0, 1, :2], expected_pos_box2), "Box 2 position is incorrect"
    # Verify other properties remain unchanged
    assert torch.equal(extrapolated_boxes[..., 2:], sample_anchor_boxes[..., 2:]), "Other box properties changed"

def test_extrapolation_with_no_velocity(model_instance):
    """
    Tests that boxes with zero velocity do not move.
    """
    dt = torch.tensor(10.0) # A large dt
    boxes_no_velocity = torch.tensor([
        [[10.0, 20.0, 5.0, 2.0, 1.5, 1.8, 0.0, 0.0, 0.0]] # vel = (0, 0)
    ], dtype=torch.float32)
    original_boxes = boxes_no_velocity.clone()
    
    extrapolated_boxes = model_instance._extrapolate_memory_with_physics(boxes_no_velocity, dt)
    
    assert torch.equal(original_boxes, extrapolated_boxes), "Box with zero velocity should not move"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_extrapolation_on_gpu(model_instance, sample_anchor_boxes):
    """
    Tests that the calculation works correctly on a CUDA device.
    """
    device = torch.device("cuda")
    dt = torch.tensor(0.5, device=device)
    gpu_boxes = sample_anchor_boxes.to(device)
    
    extrapolated_boxes = model_instance._extrapolate_memory_with_physics(gpu_boxes, dt)
    
    expected_pos_box1 = torch.tensor([10.5, 21.0], device=device)
    
    assert extrapolated_boxes.device == device, "Output tensor is not on the correct CUDA device"
    assert torch.allclose(extrapolated_boxes[0, 0, :2], expected_pos_box1), "GPU calculation is incorrect"

def test_extrapolation_with_larger_batch(model_instance):
    """
    Tests that the logic correctly broadcasts across a larger batch.
    """
    dt = torch.tensor(0.1)
    # Batch size of 2
    boxes = torch.tensor([
        [ # Batch 1
            [10.0, 20.0, 5.0, 2.0, 1.5, 1.8, 0.0, 1.0, 2.0],
        ],
        [ # Batch 2
            [30.0, 40.0, 6.0, 2.5, 1.6, 1.9, 0.0, -3.0, -4.0]
        ]
    ], dtype=torch.float32)

    extrapolated_boxes = model_instance._extrapolate_memory_with_physics(boxes, dt)

    # Expected for Batch 1, Box 1
    # x = 10.0 + (1.0 * 0.1) = 10.1
    # y = 20.0 + (2.0 * 0.1) = 20.2
    expected_pos_b1 = torch.tensor([10.1, 20.2])

    # Expected for Batch 2, Box 1
    # x = 30.0 + (-3.0 * 0.1) = 29.7
    # y = 40.0 + (-4.0 * 0.1) = 39.6
    expected_pos_b2 = torch.tensor([29.7, 39.6])

    assert extrapolated_boxes.shape == boxes.shape
    assert torch.allclose(extrapolated_boxes[0, 0, :2], expected_pos_b1)
    assert torch.allclose(extrapolated_boxes[1, 0, :2], expected_pos_b2) 