import numpy as np
import pytest
from pyquaternion import Quaternion as PyQuaternion


def compose_world_quaternion_correct(box_yaw_rad: float, ego_rot_world: PyQuaternion) -> PyQuaternion:
    """Ground-truth composition: first apply ego rotation, then box yaw."""
    box_quat = PyQuaternion(axis=[0, 0, 1], radians=box_yaw_rad)
    return ego_rot_world * box_quat  # <— correct order (as used in transforms.py)


def compose_world_quaternion_wrong(box_yaw_rad: float, ego_rot_world: PyQuaternion) -> PyQuaternion:
    """Current implementation in prediction_utils (suspected bug)."""
    box_quat = PyQuaternion(axis=[0, 0, 1], radians=box_yaw_rad)
    return box_quat * ego_rot_world  # <— wrong order


@pytest.mark.parametrize("box_yaw_deg, ego_axis, ego_angle_deg", [
    (20.0, [1, 0.2, 0.3], 15.0),  # arbitrary non-Z ego axis ⇒ order matters
    (-45.0, [0.4, 1, 0.1], 10.0),
])
def test_quaternion_multiplication_order_bug(box_yaw_deg, ego_axis, ego_angle_deg):
    """Verify that wrong order yields a different world orientation (bug still present)."""
    ego_rot = PyQuaternion(axis=ego_axis, radians=np.deg2rad(ego_angle_deg))
    yaw_rad = np.deg2rad(box_yaw_deg)

    q_correct = compose_world_quaternion_correct(yaw_rad, ego_rot)
    q_wrong = compose_world_quaternion_wrong(yaw_rad, ego_rot)

    # They should not be nearly identical (difference in elements > 1e-6).
    assert not np.allclose(q_correct.elements, q_wrong.elements, atol=1e-6), (
        "Quaternion order bug seems fixed. If you have already corrected prediction_utils,"
        " update or remove this failing test." ) 