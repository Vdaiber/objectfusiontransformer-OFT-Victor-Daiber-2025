import numpy as np
import torch
import pytest
from pyquaternion import Quaternion as PyQuaternion

# Utilities under test
from oft.transformer.datasets.truckscenes.transforms import (
    transform_world_to_ego_frame,
    transform_box_vehicle_to_world,
)
from oft.transformer.utils.normalization_utils import (
    normalize_coordinates,
    denormalize_coordinates,
    normalize_dimensions,
    denormalize_dimensions,
    get_global_normalizer,
)


def _create_dummy_pose():
    """Return a fixed ego pose (translation, rotation quaternion)."""
    translation = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    yaw = np.deg2rad(30.0)  # 30°
    rotation = PyQuaternion(axis=[0, 0, 1], radians=yaw)
    return translation, rotation


def _create_dummy_box():
    """Return a fixed 7D box in world coordinates."""
    center = np.array([10.0, 5.0, 0.5], dtype=np.float32)
    dims = np.array([4.0, 2.0, 1.5], dtype=np.float32)  # w, l, h
    yaw = np.deg2rad(45.0)  # 45°
    return np.concatenate([center, dims, np.array([yaw], dtype=np.float32)])


@pytest.mark.parametrize("seed", [0, 1])
def test_world_ego_world_roundtrip(seed):
    """world -> ego -> world should produce original box within tolerance."""
    np.random.seed(seed)
    world_box = _create_dummy_box()
    ego_t, ego_q = _create_dummy_pose()

    box_in_ego = transform_world_to_ego_frame(world_box, ego_t, ego_q)
    box_back_in_world = transform_box_vehicle_to_world(box_in_ego, ego_t, ego_q)

    np.testing.assert_allclose(
        world_box,
        box_back_in_world,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Round-trip world↔ego transform is not identity.",
    )


def test_normalization_roundtrip():
    """normalize -> denormalize should be identity for coords & dims."""
    normalizer = get_global_normalizer()
    if normalizer.stats is None:
        pytest.skip("Normalization stats file missing – cannot test round-trip.")

    point_cloud_range = torch.tensor(normalizer.stats["metadata"]["point_cloud_range"], dtype=torch.float32)

    # Use the same dummy box in ego frame (torch tensors)
    ego_box_np = _create_dummy_box()
    coords = torch.tensor(ego_box_np[:3])
    dims = torch.tensor(ego_box_np[3:6])

    coords_norm = normalize_coordinates(coords, point_cloud_range)
    coords_recon = denormalize_coordinates(coords_norm, point_cloud_range)

    torch.testing.assert_close(coords, coords_recon, rtol=1e-6, atol=1e-6)

    dims_norm = normalize_dimensions(dims)
    dims_recon = denormalize_dimensions(dims_norm)

    torch.testing.assert_close(dims, dims_recon, rtol=1e-6, atol=1e-6) 