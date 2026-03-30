import math
import numpy as np
from pyquaternion import Quaternion

from oft.transformer.utils.geometry_utils import yaw_to_sin_cos, sin_cos_to_yaw


def _quat_about_z(rad: float) -> Quaternion:
    """Create quaternion representing rotation about z-axis by *rad* radians."""
    return Quaternion(axis=[0, 0, 1], radians=rad)


# Parametrische Auswahl repräsentativer Yaw-Werte (inkl. Randfälle)
YAW_SAMPLES = [
    0.0,
    math.pi / 4,
    math.pi / 2,
    -math.pi / 2,
    math.pi - 1e-3,
    -math.pi + 1e-3,
]


def test_sin_cos_yaw_roundtrip():
    """Verify yaw → (sin,cos) → yaw is identity (≤1e-6 rad)."""
    for yaw in YAW_SAMPLES:
        s, c = yaw_to_sin_cos(yaw)
        recon = sin_cos_to_yaw(s, c)
        # Wrap to [-pi, pi] before comparison
        diff = (recon - yaw + math.pi) % (2 * math.pi) - math.pi
        assert abs(diff) < 1e-6, f"Failed roundtrip for yaw={yaw:.3f}: diff={diff}"


def test_quaternion_composition_consistency():
    """world_rotation = ego_rotation * box_quaternion should match manual composition."""
    # Random ego orientation around z and random yaw
    rng = np.random.RandomState(42)
    for _ in range(20):
        ego_yaw = rng.uniform(-math.pi, math.pi)
        box_yaw = rng.uniform(-math.pi, math.pi)

        ego_rot = _quat_about_z(ego_yaw)
        box_rot = _quat_about_z(box_yaw)

        composed = ego_rot * box_rot
        expected = _quat_about_z(ego_yaw + box_yaw)

        # Quaternions equal up to sign – compare rotation matrices
        diff = np.max(np.abs(composed.rotation_matrix - expected.rotation_matrix))
        assert diff < 1e-6, f"Quaternion composition incorrect (diff={diff})" 