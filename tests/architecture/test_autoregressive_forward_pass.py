# tests/architecture/test_autoregressive_forward_pass.py
import unittest
import torch
import yaml
from omegaconf import OmegaConf

from src.oft.transformer.models.architectures.autoregressive_architecture import ObjectFusionTransformerAutoregressive

class TestAutoregressiveForwardPass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load configuration and initialize the model once for all tests."""
        config_path = "config/pipeline_staged.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        cls.cfg = OmegaConf.create(config)
        
        # Instantiate the model
        cls.model = ObjectFusionTransformerAutoregressive(cls.cfg)
        cls.model.eval() # Set model to evaluation mode

    def _create_dummy_sensor_data(self, batch_size, num_detections, feature_dim, device):
        """Creates a dictionary of dummy data for a single sensor."""
        return {
            "features": torch.randn(batch_size, num_detections, feature_dim, device=device),
            "metadata": torch.rand(batch_size, num_detections, 2, device=device),
            "mask": torch.zeros(batch_size, num_detections, dtype=torch.bool, device=device),
        }

    def test_forward_pass_with_multiple_sensors(self):
        """
        Tests a full forward pass with dummy data from multiple sensors to verify
        the entire architecture's integrity after refactoring.
        """
        batch_size = 2
        feature_dim = self.cfg.model.input_feature_dim
        device = torch.device("cpu")
        self.model.to(device)

        # 1. Create dummy input data for all enabled sensors
        sensor_data = {}
        num_detections_map = {
            "virtual_camera": 50,
            "virtual_lidar": 100,
            "virtual_radar": 30
        }
        
        enabled_sensors = [s.name for s in self.cfg.dataset.virtual_sensors if s.enabled]
        if not enabled_sensors:
            self.skipTest("No sensors are enabled in the test configuration.")

        total_detections = 0
        for sensor_name in enabled_sensors:
            num_dets = num_detections_map.get(sensor_name, 10)
            sensor_data[sensor_name] = self._create_dummy_sensor_data(batch_size, num_dets, feature_dim, device)
            total_detections += num_dets
            
        # 2. Create dummy scene metadata
        scene_meta = [
            {"weather": "clear", "area": "highway"},
            {"weather": "rain", "area": "city"}
        ]

        # 3. Perform the forward pass
        with torch.no_grad():
            outputs, new_memory, new_memory_anchor_boxes = self.model(
                sensor_data=sensor_data,
                scene_meta=scene_meta
            )

        # 4. Verify output shapes
        self.assertIsInstance(outputs, dict)
        
        # Check decoder output shapes
        num_classes = self.cfg.model.decoder.num_classes
        num_attrs = self.cfg.model.decoder.num_attribute_classes
        
        self.assertEqual(outputs['pred_logits'].shape, (batch_size, total_detections, num_classes + 1))
        self.assertEqual(outputs['pred_box_offsets'].shape, (batch_size, total_detections, 8))
        self.assertEqual(outputs['pred_velocities'].shape, (batch_size, total_detections, 2))
        self.assertEqual(outputs['pred_attributes'].shape, (batch_size, total_detections, num_attrs + 1))
        self.assertEqual(outputs['pred_duplicate_logits'].shape, (batch_size, total_detections, 2))
        
        # Check reconstructed output shapes
        self.assertEqual(outputs['pred_boxes_normalized'].shape, (batch_size, total_detections, 8))
        self.assertEqual(outputs['pred_velocities_normalized'].shape, (batch_size, total_detections, 2))
        
        # Check memory output shapes
        fusion_d_model = self.cfg.model.inter_modal_fusion.d_model
        self.assertEqual(new_memory.shape, (batch_size, total_detections, fusion_d_model))
        self.assertEqual(new_memory_anchor_boxes.shape, (batch_size, total_detections, feature_dim))

        print("\n✅ test_forward_pass_with_multiple_sensors passed successfully.")

if __name__ == '__main__':
    unittest.main() 