import numpy as np
import pytest
from pyquaternion import Quaternion
from src.oft.transformer.datasets.truckscenes.transforms import (
    transform_box_vehicle_to_world,
    transform_world_to_ego_frame
)

def make_random_box():
    center = np.random.uniform(-10, 10, size=3)
    wlh = np.random.uniform(1, 5, size=3)
    yaw = np.random.uniform(-np.pi, np.pi)
    return np.array([*center, *wlh, yaw], dtype=np.float32)

def make_random_pose():
    translation = np.random.uniform(-50, 50, size=3)
    angle = np.random.uniform(-np.pi, np.pi)
    rotation = Quaternion(axis=[0,0,1], angle=angle)
    return translation, rotation

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_vehicle_world_roundtrip(seed):
    np.random.seed(seed)
    for _ in range(10):
        box_ego = make_random_box()
        ego_translation, ego_rotation = make_random_pose()
        box_world = transform_box_vehicle_to_world(box_ego, ego_translation, ego_rotation)
        box_ego_back = transform_world_to_ego_frame(box_world, ego_translation, ego_rotation)
        assert np.allclose(box_ego, box_ego_back, atol=1e-5), f"Roundtrip failed: {box_ego} vs {box_ego_back}"

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__])) 