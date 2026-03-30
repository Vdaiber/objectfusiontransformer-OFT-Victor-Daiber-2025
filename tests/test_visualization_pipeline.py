import pytest
import numpy as np
from pyquaternion import Quaternion
from unittest.mock import MagicMock, patch
from oft.transformer.utils.epoch_visualization_workflow import visualize_val_sample_from_dataset

class DummyBox:
    def __init__(self, center, size, orientation, name, score, token):
        self.center = center
        self.size = size
        self.orientation = orientation
        self.name = name
        self.score = score
        self.token = token
    def render_cv2(self, img, view, normalize, colors, linewidth):
        img[0,0,0] = 255

@pytest.fixture
def dummy_val_dataset():
    # Simuliere ein val_dataset mit allen nötigen Feldern für die Visualisierung
    class DummyValDataset:
        def __getitem__(self, idx):
            return {
                'sample_token': 'dummy_token',
                'ego_translation_world': np.array([1.0,2.0,3.0], dtype=np.float32),
                'ego_rotation_world_quat': np.array([1.0,0.0,0.0,0.0], dtype=np.float32)
            }
        def _get_full_box_data_for_token_world(self, token):
            return [{
                'box_7d_world': np.array([1,2,3,4,5,6,0.1], dtype=np.float32),
                'detection_name': 'vehicle.car'
            }]
        class ts:
            dataroot = '/tmp'
            @staticmethod
            def get(kind, token):
                if kind == 'sample':
                    return {'data': {'CAMERA_LEFT_FRONT': 'cam_token'}}
                if kind == 'sample_data':
                    return {'calibrated_sensor_token': 'calib_token', 'filename': 'dummy.jpg'}
                if kind == 'calibrated_sensor':
                    return {'camera_intrinsic': [1.0]*9, 'rotation': [1.0,0.0,0.0,0.0], 'translation': [0.0,0.0,0.0]}
                return {}
    return DummyValDataset()

@pytest.fixture
def dummy_cfg():
    return {
        'visualization': {'sample_idx': 0, 'camera_channel': 'CAMERA_LEFT_FRONT'},
        'dataset': {'class_names': ['car','truck','pedestrian','traffic_sign']}
    }

@pytest.fixture
def dummy_predicted_boxes():
    return [np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.0,1.0,0], dtype=np.float32),
            np.array([0.2,0.3,0.4,0.5,0.6,0.7,1.0,0.0,2], dtype=np.float32)]

@patch('oft.transformer.utils.epoch_visualization_workflow.DevkitBox', DummyBox)
@patch('oft.transformer.utils.epoch_visualization_workflow.visualize_epoch_sample')
@patch('cv2.imread', return_value=np.ones((100,100,3), dtype=np.uint8))
def test_visualization_pipeline(mock_imread, mock_vis, dummy_cfg, dummy_val_dataset, dummy_predicted_boxes):
    visualize_val_sample_from_dataset(
        cfg=dummy_cfg,
        val_dataset=dummy_val_dataset,
        epoch=0,
        run_dir="/tmp",
        trucksc=None,
        predicted_boxes_normalized=dummy_predicted_boxes
    )
    assert mock_vis.called, "Visualisierung wurde nicht aufgerufen!"
    call_args = mock_vis.call_args
    kwargs = call_args[1]
    fused_boxes = kwargs.get('fused_boxes', [])
    gt_boxes = kwargs.get('gt_boxes', [])
    assert len(fused_boxes) > 0, "Keine Predicted-Boxen übergeben!"
    assert len(gt_boxes) > 0, "Keine GT-Boxen übergeben!"
    for box in fused_boxes+gt_boxes:
        assert isinstance(box.center, np.ndarray)
        assert box.center.shape == (3,)
    for box in gt_boxes:
        assert box.name.startswith('vehicle.') or box.name.startswith('human.') or box.name in ['animal','static_object.traffic_sign','movable_object.trafficcone','movable_object.barrier']
    assert mock_imread.called 