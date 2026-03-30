import numpy as np
import torch

from oft.transformer.datasets.preprocessing.data_augmentation import add_attribute_noise
from oft.transformer.scripts.evaluate_virtual_sensors_baseline import VirtualSensorBaselineEvaluator
from oft.transformer.evaluation.prediction_utils import reconstruct_and_convert_predictions_autoregressive


def test_add_attribute_noise_preserves_background():
    rng = np.random.default_rng(123)
    attribute_vocab = list(range(11))  # 0..10 are valid attributes
    sensor_cfg = {"name": "virtual_camera", "class_accuracy": 0.9, "attribute_accuracy": 0.9}
    # background index is len(attribute_vocab) == 11
    out = add_attribute_noise(11, sensor_cfg, attribute_vocab, rng=rng)
    assert out == 11


def test_baseline_conversion_maps_background_to_empty(monkeypatch, tmp_path):
    # Build a minimal evaluator with a tiny config
    cfg = {
        'dataset': {
            'version': 'v1.0-mini',
            'dataroot': '/data',
            'class_names': ['car'],
            'virtual_sensors': [{'name': 'virtual_camera', 'enabled': True}],
        },
        'evaluation': {'baseline_eval_split': 'mini_val'},
        'training': {'val_split_name': 'mini_val', 'batch_size': 1},
    }

    # Instantiate evaluator object without reading files
    ev = VirtualSensorBaselineEvaluator.__new__(VirtualSensorBaselineEvaluator)
    ev.config_path = ''
    ev.config = cfg
    ev.output_dir = str(tmp_path)
    ev.logger = type('L', (), {'info': lambda *a, **k: None, 'warning': lambda *a, **k: None, 'error': lambda *a, **k: None})()

    # Prepare inputs for _convert_raw_box_to_devkit_format
    import numpy as np
    box_9d = torch.tensor([0,0,0, 1,2,3, 0.0, 0.0, 0.0], dtype=torch.float64)
    # features_12d indices: ... last two are [sensor_class, sensor_attr]
    features_12d = torch.zeros(12, dtype=torch.float64)
    features_12d[10] = 0  # class car
    features_12d[11] = 11 # background attribute

    # Fake batch_dict ego pose
    batch_dict = {
        'ego_translation_world': torch.tensor([[0.0,0.0,0.0]], dtype=torch.float64),
        'ego_rotation_world_quat': torch.tensor([[1.0,0.0,0.0,0.0]], dtype=torch.float64)
    }

    pred = ev._convert_raw_box_to_devkit_format(box_9d, features_12d, batch_dict, 0, 'virtual_camera', 'tok')
    assert pred is not None
    assert pred['attribute_name'] == ''


def test_prediction_utils_background_to_empty():
    B, N, C = 1, 2, 13  # num_classes+1 not used here; attributes handled separately
    batch = {
        'sample_tokens': ['tok'],
        'ego_translation_world': torch.zeros(B, 3, dtype=torch.float64),
        'ego_rotation_world_quat': torch.tensor([[1.0,0.0,0.0,0.0]], dtype=torch.float64),
        'ego_motion': {'cabin': {'velocity': torch.zeros(B,3, dtype=torch.float64)}},
    }
    predictions = {
        'pred_class_logits_batch': torch.zeros(B, N, 13),
        'pred_boxes_normalized': torch.zeros(B, N, 10, dtype=torch.float64),
        'pred_velocities_normalized': torch.zeros(B, N, 2, dtype=torch.float64),
        'pred_attributes_logits_batch': torch.zeros(B, N, 12)  # 11 + background
    }
    # Force attributes argmax to background 11
    predictions['pred_attributes_logits_batch'][0, :, 11] = 10.0
    cfg = {'evaluation': {'conf_th_eval': 0.0}, 'dataset': {'class_names': ['car']}}
    out = reconstruct_and_convert_predictions_autoregressive(batch, predictions, cfg)
    # Some predictions may be filtered by class; ensure that any attribute present is empty
    for item in out:
        for pred in item['predictions']:
            assert pred.get('attribute_name', '') in ('',)


