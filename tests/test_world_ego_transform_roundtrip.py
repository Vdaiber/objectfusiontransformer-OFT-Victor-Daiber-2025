import numpy as np
import math
from pyquaternion import Quaternion as PyQuaternion

from oft.transformer.datasets.truckscenes.transforms import (
    transform_world_to_ego_frame,
    transform_box_vehicle_to_world,
)


def _random_box_7d_world(rng: np.random.RandomState):
    center = rng.uniform([-50, -50, -3], [50, 50, 3])  # xyz
    dims = rng.uniform([1.5, 3.0, 1.5], [2.5, 5.0, 3.0])  # w,l,h
    yaw = rng.uniform(-math.pi, math.pi)
    return np.concatenate([center, dims, [yaw]]).astype(np.float32)


def test_world_ego_world_roundtrip():
    rng = np.random.RandomState(0)
    for _ in range(30):
        box_world = _random_box_7d_world(rng)
        ego_trans = rng.uniform([-10, -10, -1], [10, 10, 1]).astype(np.float32)
        ego_yaw = rng.uniform(-math.pi, math.pi)
        ego_rot = PyQuaternion(axis=[0, 0, 1], radians=ego_yaw)

        # world → ego → world
        box_ego = transform_world_to_ego_frame(box_world, ego_trans, ego_rot)
        box_world_recon = transform_box_vehicle_to_world(box_ego, ego_trans, ego_rot)

        diff = np.abs(box_world - box_world_recon)
        # For angle: account for wrap-around
        diff[6] = ((box_world_recon[6] - box_world[6] + math.pi) % (2 * math.pi)) - math.pi
        assert diff[:6].max() < 1e-5, f"World↔Ego center/dims mismatch {diff[:6].max()}m"
        assert abs(diff[6]) < 1e-5, f"Yaw mismatch {diff[6]} rad" 