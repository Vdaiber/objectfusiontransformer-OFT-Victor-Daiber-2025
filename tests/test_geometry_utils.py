"""
Tests for geometry utilities.

This module tests the geometry utility functions including yaw angle conversions
and scene description parsing.
"""

import pytest
import numpy as np
import math
from oft.transformer.utils.geometry_utils import yaw_to_sin_cos, sin_cos_to_yaw, parse_scene_description


class TestYawConversions:
    """Test yaw angle conversion functions."""
    
    def test_yaw_to_sin_cos_basic(self):
        """Test basic yaw to sin/cos conversion."""
        # Test 0 degrees
        sin_yaw, cos_yaw = yaw_to_sin_cos(0.0)
        assert abs(sin_yaw - 0.0) < 1e-6
        assert abs(cos_yaw - 1.0) < 1e-6
        
        # Test 90 degrees (π/2 radians)
        sin_yaw, cos_yaw = yaw_to_sin_cos(math.pi / 2)
        assert abs(sin_yaw - 1.0) < 1e-6
        assert abs(cos_yaw - 0.0) < 1e-6
        
        # Test 180 degrees (π radians)
        sin_yaw, cos_yaw = yaw_to_sin_cos(math.pi)
        assert abs(sin_yaw - 0.0) < 1e-6
        assert abs(cos_yaw - (-1.0)) < 1e-6
        
        # Test 270 degrees (3π/2 radians)
        sin_yaw, cos_yaw = yaw_to_sin_cos(3 * math.pi / 2)
        assert abs(sin_yaw - (-1.0)) < 1e-6
        assert abs(cos_yaw - 0.0) < 1e-6
    
    def test_sin_cos_to_yaw_basic(self):
        """Test basic sin/cos to yaw conversion."""
        # Test 0 degrees
        yaw = sin_cos_to_yaw(0.0, 1.0)
        assert abs(yaw - 0.0) < 1e-6
        
        # Test 90 degrees
        yaw = sin_cos_to_yaw(1.0, 0.0)
        assert abs(yaw - math.pi / 2) < 1e-6
        
        # Test 180 degrees
        yaw = sin_cos_to_yaw(0.0, -1.0)
        assert abs(yaw - math.pi) < 1e-6
        
        # Test 270 degrees
        yaw = sin_cos_to_yaw(-1.0, 0.0)
        assert abs(yaw - (-math.pi / 2)) < 1e-6
    
    def test_yaw_conversion_roundtrip(self):
        """Test that yaw conversion is reversible."""
        test_angles = [-math.pi, -math.pi/2, 0, math.pi/2, math.pi, 1.5, -0.7]
        
        for angle in test_angles:
            sin_yaw, cos_yaw = yaw_to_sin_cos(angle)
            converted_angle = sin_cos_to_yaw(sin_yaw, cos_yaw)
            
            # Handle angle wrapping (atan2 returns [-π, π])
            angle_diff = abs(converted_angle - angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            
            assert angle_diff < 1e-6, f"Failed for angle {angle}"
    
    def test_yaw_boundary_conditions(self):
        """Test yaw conversion at boundary conditions."""
        # Test at π and -π boundary
        sin_yaw, cos_yaw = yaw_to_sin_cos(math.pi)
        yaw_back = sin_cos_to_yaw(sin_yaw, cos_yaw)
        assert abs(yaw_back - math.pi) < 1e-6
        
        sin_yaw, cos_yaw = yaw_to_sin_cos(-math.pi)
        yaw_back = sin_cos_to_yaw(sin_yaw, cos_yaw)
        assert abs(yaw_back - (-math.pi)) < 1e-6


class TestSceneDescriptionParsing:
    """Test scene description parsing function."""
    
    def test_parse_scene_description_basic(self):
        """Test basic scene description parsing."""
        description = "weather.clear;area.highway;time.day"
        result = parse_scene_description(description)
        
        expected = {
            "weather": "clear",
            "area": "highway", 
            "time": "day"
        }
        assert result == expected
    
    def test_parse_scene_description_empty(self):
        """Test parsing empty description."""
        result = parse_scene_description("")
        assert result == {}
        
        result = parse_scene_description(None)
        assert result == {}
    
    def test_parse_scene_description_complex(self):
        """Test parsing complex scene description."""
        description = "weather.rain;area.urban;time.night;season.winter;lighting.street_lights"
        result = parse_scene_description(description)
        
        expected = {
            "weather": "rain",
            "area": "urban",
            "time": "night", 
            "season": "winter",
            "lighting": "street_lights"
        }
        assert result == expected
    
    def test_parse_scene_description_malformed(self):
        """Test parsing malformed description."""
        # Missing dot
        description = "weather.clear;area;time.day"
        result = parse_scene_description(description)
        expected = {"weather": "clear", "time": "day"}
        assert result == expected
        
        # Empty parts
        description = "weather.clear;;time.day;"
        result = parse_scene_description(description)
        expected = {"weather": "clear", "time": "day"}
        assert result == expected
    
    def test_parse_scene_description_special_characters(self):
        """Test parsing description with special characters."""
        description = "weather.clear;area.highway_construction;time.day"
        result = parse_scene_description(description)
        
        expected = {
            "weather": "clear",
            "area": "highway_construction",
            "time": "day"
        }
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__]) 