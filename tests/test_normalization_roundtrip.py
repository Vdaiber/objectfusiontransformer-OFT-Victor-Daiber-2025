import torch

from oft.transformer.utils.normalization_utils import (
    normalize_coordinates,
    denormalize_coordinates,
    normalize_dimensions,
    denormalize_dimensions,
)


def test_normalization_roundtrip():
    """Ensure that normalize→denormalize yields original values (within 1 mm)."""
    point_cloud_range = torch.tensor([-149.65, -146.07, -8.61, 149.75, 149.43, 11.73])

    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [10.0, -5.0, 1.0],
            [120.0, 30.0, 5.0],
        ]
    )
    dims = torch.tensor(
        [
            [1.6, 4.0, 1.5],
            [2.2, 5.5, 2.0],
            [0.5, 0.5, 1.0],
        ]
    )

    roundtrip_coords = denormalize_coordinates(
        normalize_coordinates(coords, point_cloud_range), point_cloud_range
    )
    roundtrip_dims = denormalize_dimensions(normalize_dimensions(dims))

    assert torch.allclose(coords, roundtrip_coords, atol=1e-3)
    assert torch.allclose(dims, roundtrip_dims, atol=1e-3) 