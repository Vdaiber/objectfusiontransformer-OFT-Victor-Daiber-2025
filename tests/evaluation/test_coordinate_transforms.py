# tests/evaluation/test_coordinate_transforms.py
import unittest
import numpy as np
from pyquaternion import Quaternion
import sys
import os

from src.oft.datasets.truckscenes.builder import Box

def transform_ego_to_world_buggy(box_in_ego: Box, ego_translation_world: np.ndarray, ego_rotation_world: Quaternion) -> Box:
    """
    This function replicates the CURRENTLY FLAWED transformation logic found in the evaluator.
    It incorrectly rotates the box around the ego vehicle's origin instead of its own center.
    WE EXPECT THIS TEST TO FAIL.
    """
    box_to_transform = box_in_ego.copy()
    
    # This is the flawed logic:
    # 1. It rotates the box's center point around the EGO ORIGIN (0,0,0)
    box_to_transform.rotate(ego_rotation_world)
    # 2. It then translates the already incorrectly rotated point.
    box_to_transform.translate(ego_translation_world)
    
    return box_to_transform

def transform_ego_to_world_correct(box_in_ego: Box, ego_translation_world: np.ndarray, ego_rotation_world: Quaternion) -> Box:
    """
    This function implements the MATHEMATICALLY CORRECT transformation from ego to world frame.
    WE EXPECT THIS TEST TO SUCCEED.
    """
    box_to_transform = box_in_ego.copy()
    
    # Correct logic:
    # 1. First, rotate the box around its OWN center to align it with the world axes.
    box_to_transform.rotate(ego_rotation_world)
    
    # 2. Then, translate the box by the ego vehicle's translation vector.
    # Note: The nuscenes Box class handles this correctly internally.
    # The rotate() method rotates the orientation and also moves the center.
    # The translate() method then adds the translation.
    # Let's do it manually to be explicit.

    # Manual, explicit transformation:
    # a. Rotate the center point of the box by the ego rotation
    new_center = ego_rotation_world.rotate(box_in_ego.center)
    # b. Add the ego vehicle's world translation
    new_center += ego_translation_world
    
    # c. The new orientation is the ego orientation multiplied by the box's orientation
    new_orientation = ego_rotation_world * box_in_ego.orientation

    return Box(new_center, box_in_ego.wlh, new_orientation)


class TestCoordinateTransforms(unittest.TestCase):

    def test_ego_to_world_transformation(self):
        """
        Validates the ego-to-world coordinate transformation by comparing the flawed
        implementation with the correct one against a known ground truth.
        """
        # 1. GIVEN: Define the scenario
        # Ego vehicle is at (10, 20, 0) in the world, rotated by +90 degrees around Z.
        ego_translation = np.array([10, 20, 0])
        ego_rotation = Quaternion(axis=[0, 0, 1], degrees=90) # w, x, y, z

        # A box is 5 meters straight ahead of the ego vehicle, with no rotation in the ego frame.
        box_center_ego = [5, 0, 0]
        box_size = [4, 2, 1.5] # l, w, h
        box_orientation_ego = Quaternion(axis=[0, 0, 1], degrees=0)
        box_in_ego = Box(box_center_ego, box_size, box_orientation_ego)

        # 2. EXPECTED: Calculate the correct world position and orientation
        # The box should be at (10, 25, 0) in world coordinates with a 90-degree rotation.
        expected_center_world = np.array([10, 25, 0])
        expected_orientation_world = ego_rotation * box_orientation_ego

        # 3. WHEN: Run both the buggy and the correct transformation
        result_buggy = transform_ego_to_world_buggy(box_in_ego, ego_translation, ego_rotation)
        result_correct = transform_ego_to_world_correct(box_in_ego, ego_translation, ego_rotation)
        
        print("\n--- Coordinate Transformation Test ---")
        print(f"Ego Pose: T={ego_translation}, R={ego_rotation.degrees}°")
        print(f"Box in Ego Frame: Center={box_in_ego.center}")
        print("-" * 34)
        print(f"EXPECTED World Center: {expected_center_world}")
        print(f"BUGGY Result Center:   {result_buggy.center}")
        print(f"CORRECT Result Center: {result_correct.center}")
        print("-" * 34)

        # 4. THEN: Assert the results
        # Assert that the buggy version is NOT correct
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(result_buggy.center, expected_center_world, atol=1e-5)
        
        # Assert that the correct version IS correct
        np.testing.assert_allclose(result_correct.center, expected_center_world, atol=1e-5)
        self.assertTrue(result_correct.orientation.is_close(expected_orientation_world))
        
        print("✅ Test confirms: Buggy transformation is incorrect, correct transformation is valid.")

if __name__ == '__main__':
    unittest.main() 