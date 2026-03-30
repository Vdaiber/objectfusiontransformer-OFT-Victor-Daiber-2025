#!/usr/bin/env python3
"""
Comprehensive analysis of visualization box filtering pipeline.
Analyzes why only 11 out of 14 GT boxes and 11 out of 30+ predicted boxes are visible.

This test investigates:
1. FOV filtering logic (_filter_boxes_in_camera_fov)
2. Timestamp consistency between dataset and DevKit
3. Coordinate transformations (World → Ego → Camera)
4. Specific sample predictions for 'f1c03220990143e19b983bf3da478764'
"""

import os
import sys
import json
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from pyquaternion import Quaternion

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from truckscenes import TruckScenes
from truckscenes.utils.data_classes import Box as DevkitBox
from truckscenes.utils.geometry_utils import BoxVisibility
from oft.transformer.utils.epoch_visualization_workflow import _filter_boxes_in_camera_fov


class VisualizationBoxAnalyzer:
    """Analyzer for visualization box filtering pipeline."""
    
    def __init__(self, dataroot: str = "/data", version: str = "v1.0-mini"):
        """Initialize analyzer with TruckScenes dataset."""
        self.ts = TruckScenes(version=version, dataroot=dataroot, verbose=False)
        self.test_sample_token = "f1c03220990143e19b983bf3da478764"
        self.camera_channel = "CAMERA_LEFT_FRONT"
        
    def analyze_latest_predictions(self, run_dir: str, epoch: int = 13) -> Dict[str, Any]:
        """Analyze predictions from latest run."""
        pred_path = os.path.join(run_dir, 'eval_results', f'epoch_{epoch}', 'predictions.json')
        
        if not os.path.exists(pred_path):
            return {"error": f"Predictions file not found: {pred_path}"}
            
        with open(pred_path, 'r') as f:
            predictions = json.load(f)
            
        sample_predictions = predictions['results'].get(self.test_sample_token, [])
        
        analysis = {
            "predictions_file": pred_path,
            "total_samples": len(predictions['results']),
            "test_sample_predictions": len(sample_predictions),
            "sample_predictions": sample_predictions
        }
        
        return analysis
        
    def analyze_gt_boxes(self) -> Dict[str, Any]:
        """Analyze GT boxes for the test sample."""
        sample_record = self.ts.get('sample', self.test_sample_token)
        cam_token = sample_record['data'][self.camera_channel]
        
        # Get GT boxes using DevKit standard method
        _, gt_boxes_cam, _ = self.ts.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
        
        # Get all sample annotations for this sample
        all_annotations = []
        for ann_token in sample_record['anns']:
            ann_record = self.ts.get('sample_annotation', ann_token)
            all_annotations.append(ann_record)
            
        analysis = {
            "total_annotations": len(all_annotations),
            "visible_gt_boxes": len(gt_boxes_cam),
            "annotation_details": [],
            "gt_box_details": []
        }
        
        # Analyze each annotation
        for i, ann in enumerate(all_annotations):
            details = {
                "index": i,
                "category": ann['category_name'],
                "translation": ann['translation'],
                "size": ann['size'],
                "visibility": ann.get('visibility_token', 'unknown')
            }
            analysis["annotation_details"].append(details)
            
        # Analyze each visible GT box
        for i, box in enumerate(gt_boxes_cam):
            details = {
                "index": i,
                "name": box.name,
                "center": box.center.tolist(),
                "size": box.wlh.tolist(),
                "score": getattr(box, 'score', None)
            }
            analysis["gt_box_details"].append(details)
            
        return analysis
        
    def analyze_coordinate_transforms(self, sample_predictions: List[Dict]) -> Dict[str, Any]:
        """Analyze coordinate transformations for predicted boxes."""
        if not sample_predictions:
            return {"error": "No sample predictions provided"}
            
        sample_record = self.ts.get('sample', self.test_sample_token)
        cam_token = sample_record['data'][self.camera_channel]
        sd_record = self.ts.get('sample_data', cam_token)
        cs_record = self.ts.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        
        # Get consistent ego pose (same method as dataset)
        pose_record = self.ts.getclosest('ego_pose', sample_record['timestamp'])
        
        analysis = {
            "timestamp_consistency": {
                "sample_timestamp": sample_record['timestamp'],
                "ego_pose_timestamp": pose_record['timestamp'],
                "timestamp_diff_ms": abs(sample_record['timestamp'] - pose_record['timestamp']) / 1000.0
            },
            "transformations": [],
            "world_boxes": [],
            "ego_boxes": [],
            "camera_boxes": []
        }
        
        for i, pred in enumerate(sample_predictions):
            # Create world box
            center_world = np.array(pred['translation'])
            size_world = np.array(pred['size'])
            rotation_world_quat = Quaternion(pred['rotation'])
            
            world_box = DevkitBox(
                center=center_world,
                size=size_world,
                orientation=rotation_world_quat,
                name=pred['detection_name'],
                score=pred['detection_score'],
                token=f'pred_{i}'
            )
            
            # Transform: World → Ego
            ego_box = world_box.copy()
            ego_box.translate(-np.array(pose_record['translation']))
            ego_box.rotate(Quaternion(pose_record['rotation']).inverse)
            
            # Transform: Ego → Camera
            camera_box = ego_box.copy()
            camera_box.translate(-np.array(cs_record['translation']))
            camera_box.rotate(Quaternion(cs_record['rotation']).inverse)
            
            transform_details = {
                "index": i,
                "detection_name": pred['detection_name'],
                "detection_score": pred['detection_score'],
                "world_center": center_world.tolist(),
                "ego_center": ego_box.center.tolist(),
                "camera_center": camera_box.center.tolist(),
                "camera_z_depth": camera_box.center[2]  # Critical for FOV filtering
            }
            
            analysis["transformations"].append(transform_details)
            analysis["world_boxes"].append(world_box)
            analysis["ego_boxes"].append(ego_box)
            analysis["camera_boxes"].append(camera_box)
            
        return analysis
        
    def analyze_fov_filtering(self, camera_boxes: List[DevkitBox]) -> Dict[str, Any]:
        """Analyze FOV filtering step by step."""
        if not camera_boxes:
            return {"error": "No camera boxes provided"}
            
        sample_record = self.ts.get('sample', self.test_sample_token)
        cam_token = sample_record['data'][self.camera_channel]
        sd_record = self.ts.get('sample_data', cam_token)
        cs_record = self.ts.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        
        # Load camera intrinsics and image
        K = np.array(cs_record['camera_intrinsic'], dtype=np.float64).reshape(3, 3)
        image_path = os.path.join(self.ts.dataroot, sd_record['filename'])
        
        # Get image shape (we don't need to load the actual image for analysis)
        import cv2
        image = cv2.imread(image_path)
        img_shape = image.shape
        h, w = img_shape[:2]
        
        analysis = {
            "image_shape": img_shape,
            "camera_intrinsics": K.tolist(),
            "total_boxes": len(camera_boxes),
            "box_analysis": [],
            "filtering_results": {
                "depth_failed": 0,
                "projection_failed": 0,
                "visibility_failed": 0,
                "passed": 0
            }
        }
        
        visible_boxes = []
        
        for i, box in enumerate(camera_boxes):
            # Get box corners in camera coordinates
            corners_cam = box.corners()  # 3x8 array
            
            # Depth check
            min_depth = np.min(corners_cam[2, :])
            max_depth = np.max(corners_cam[2, :])
            depth_pass = np.all(corners_cam[2, :] > 0)
            
            # Projection to image
            homog_pts = K @ corners_cam  # 3x8
            u = homog_pts[0, :] / homog_pts[2, :]
            v = homog_pts[1, :] / homog_pts[2, :]
            
            # Visibility check
            in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            visibility_pass = np.any(in_img)
            
            # Overall pass/fail
            overall_pass = depth_pass and visibility_pass
            
            box_details = {
                "index": i,
                "name": box.name,
                "score": getattr(box, 'score', None),
                "center": box.center.tolist(),
                "depth_range": [float(min_depth), float(max_depth)],
                "depth_pass": bool(depth_pass),
                "projection_u_range": [float(np.min(u)), float(np.max(u))],
                "projection_v_range": [float(np.min(v)), float(np.max(v))],
                "pixels_in_image": int(np.sum(in_img)),
                "total_corners": 8,
                "visibility_pass": bool(visibility_pass),
                "overall_pass": bool(overall_pass)
            }
            
            analysis["box_analysis"].append(box_details)
            
            # Update filtering results
            if not depth_pass:
                analysis["filtering_results"]["depth_failed"] += 1
            elif not visibility_pass:
                analysis["filtering_results"]["visibility_failed"] += 1
            else:
                analysis["filtering_results"]["passed"] += 1
                visible_boxes.append(box)
                
        # Verify with actual filtering function
        actual_filtered = _filter_boxes_in_camera_fov(camera_boxes, K, img_shape)
        
        analysis["verification"] = {
            "manual_count": len(visible_boxes),
            "function_count": len(actual_filtered),
            "counts_match": len(visible_boxes) == len(actual_filtered)
        }
        
        return analysis
        
    def run_complete_analysis(self, run_dir: str, epoch: int = 13) -> Dict[str, Any]:
        """Run complete analysis of visualization pipeline."""
        print(f"\n🔍 COMPREHENSIVE VISUALIZATION ANALYSIS")
        print(f"Sample Token: {self.test_sample_token}")
        print(f"Camera: {self.camera_channel}")
        print("=" * 80)
        
        # 1. Analyze predictions
        print("\n1. ANALYZING PREDICTIONS...")
        pred_analysis = self.analyze_latest_predictions(run_dir, epoch)
        
        # 2. Analyze GT boxes
        print("\n2. ANALYZING GT BOXES...")
        gt_analysis = self.analyze_gt_boxes()
        
        # 3. Analyze coordinate transformations
        print("\n3. ANALYZING COORDINATE TRANSFORMATIONS...")
        if 'sample_predictions' in pred_analysis:
            transform_analysis = self.analyze_coordinate_transforms(pred_analysis['sample_predictions'])
        else:
            transform_analysis = {"error": "No predictions to analyze"}
            
        # 4. Analyze FOV filtering
        print("\n4. ANALYZING FOV FILTERING...")
        if 'camera_boxes' in transform_analysis:
            fov_analysis = self.analyze_fov_filtering(transform_analysis['camera_boxes'])
        else:
            fov_analysis = {"error": "No camera boxes to analyze"}
            
        # Compile complete analysis
        complete_analysis = {
            "test_sample": self.test_sample_token,
            "camera_channel": self.camera_channel,
            "predictions": pred_analysis,
            "ground_truth": gt_analysis,
            "transformations": transform_analysis,
            "fov_filtering": fov_analysis
        }
        
        return complete_analysis
        
    def print_analysis_summary(self, analysis: Dict[str, Any]):
        """Print human-readable summary of analysis."""
        print("\n" + "="*80)
        print("📊 ANALYSIS SUMMARY")
        print("="*80)
        
        # Predictions summary
        if 'predictions' in analysis:
            pred = analysis['predictions']
            print(f"\n🎯 PREDICTIONS:")
            print(f"   • Total samples in file: {pred.get('total_samples', 'N/A')}")
            print(f"   • Predictions for test sample: {pred.get('test_sample_predictions', 'N/A')}")
            
        # GT summary
        if 'ground_truth' in analysis:
            gt = analysis['ground_truth']
            print(f"\n📍 GROUND TRUTH:")
            print(f"   • Total annotations: {gt.get('total_annotations', 'N/A')}")
            print(f"   • Visible GT boxes: {gt.get('visible_gt_boxes', 'N/A')}")
            print(f"   • Missing boxes: {gt.get('total_annotations', 0) - gt.get('visible_gt_boxes', 0)}")
            
        # Transformations summary
        if 'transformations' in analysis:
            trans = analysis['transformations']
            if 'timestamp_consistency' in trans:
                ts = trans['timestamp_consistency']
                print(f"\n🕒 TIMESTAMP CONSISTENCY:")
                print(f"   • Timestamp difference: {ts.get('timestamp_diff_ms', 'N/A')} ms")
                
        # FOV filtering summary
        if 'fov_filtering' in analysis:
            fov = analysis['fov_filtering']
            if 'filtering_results' in fov:
                results = fov['filtering_results']
                print(f"\n🔍 FOV FILTERING RESULTS:")
                print(f"   • Total boxes: {fov.get('total_boxes', 'N/A')}")
                print(f"   • Depth failed: {results.get('depth_failed', 'N/A')}")
                print(f"   • Visibility failed: {results.get('visibility_failed', 'N/A')}")
                print(f"   • Passed: {results.get('passed', 'N/A')}")
                
        print("\n" + "="*80)


def test_visualization_box_filtering():
    """Main test function."""
    # Use actual run directory from logs
    run_dir = "/data/daiber_fent/output/staged_runs/2025-08-09/09-36-01"
    
    analyzer = VisualizationBoxAnalyzer()
    
    # Run complete analysis
    analysis = analyzer.run_complete_analysis(run_dir, epoch=13)
    
    # Print summary
    analyzer.print_analysis_summary(analysis)
    
    # Save detailed analysis to file
    output_file = "tests/visualization_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\n💾 Detailed analysis saved to: {output_file}")
    
    return analysis


if __name__ == "__main__":
    test_visualization_box_filtering()
