"""
YOLOv8 Comparison Module
Implements YOLOv8-based detection for comparison with classical methods
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class YOLOv8Result:
    """Container for YOLOv8 detection results"""
    count: int
    bboxes: List[Tuple[int, int, int, int]]
    confidence_scores: List[float]
    class_ids: List[int]
    class_names: List[str]


class YOLOv8Comparator:
    """
    YOLOv8-based detector for comparing deep learning vs classical approaches.
    
    This class provides a wrapper around YOLOv8 for:
    - Plant detection and counting
    - Performance comparison with classical methods
    - Benchmark evaluation on ornamental species
    
    Args:
        model_path: Path to YOLOv8 model weights (optional)
        confidence_threshold: Minimum confidence for detection (default: 0.5)
        iou_threshold: IoU threshold for NMS (default: 0.45)
        device: Device for inference ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = 'cpu'
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """
        Load YOLOv8 model from weights file.
        Deferred implementation - requires ultralytics package.
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
        except ImportError:
            print("Warning: ultralytics not installed. Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def detect(
        self, 
        image: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> YOLOv8Result:
        """
        Detect plants using YOLOv8.
        
        Args:
            image: Input image (BGR format)
            classes: Optional list of class IDs to detect
            
        Returns:
            YOLOv8Result containing detections
        """
        if self.model is None:
            # Return empty result if model not loaded
            return YOLOv8Result(
                count=0,
                bboxes=[],
                confidence_scores=[],
                class_ids=[],
                class_names=[]
            )
        
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=classes,
            verbose=False
        )
        
        # Parse results
        return self._parse_results(results[0])
    
    def _parse_results(self, result) -> YOLOv8Result:
        """Parse YOLOv8 detection results"""
        bboxes = []
        scores = []
        class_ids = []
        class_names = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))
                scores.append(float(conf))
                class_ids.append(int(cls))
                class_names.append(result.names[cls])
        
        return YOLOv8Result(
            count=len(bboxes),
            bboxes=bboxes,
            confidence_scores=scores,
            class_ids=class_ids,
            class_names=class_names
        )
    
    def compare_with_classical(
        self,
        image: np.ndarray,
        classical_result: Dict
    ) -> Dict[str, any]:
        """
        Compare YOLOv8 results with classical method results.
        
        Args:
            image: Input image
            classical_result: Results from classical detection method
            
        Returns:
            Dictionary containing comparison metrics
        """
        yolo_result = self.detect(image)
        
        comparison = {
            'yolov8_count': yolo_result.count,
            'classical_count': classical_result.get('count', 0),
            'count_difference': abs(yolo_result.count - classical_result.get('count', 0)),
            'yolov8_confidence_mean': np.mean(yolo_result.confidence_scores) if yolo_result.confidence_scores else 0,
            'yolov8_confidence_std': np.std(yolo_result.confidence_scores) if yolo_result.confidence_scores else 0,
        }
        
        # IoU-based matching if both have detections
        if yolo_result.bboxes and classical_result.get('bboxes'):
            iou_matrix = self._compute_iou_matrix(
                yolo_result.bboxes,
                classical_result['bboxes']
            )
            comparison['mean_iou'] = np.mean(iou_matrix[iou_matrix > 0.1])
            comparison['matched_detections'] = np.sum(np.max(iou_matrix, axis=1) > 0.5)
        else:
            comparison['mean_iou'] = 0.0
            comparison['matched_detections'] = 0
        
        return comparison
    
    def _compute_iou_matrix(
        self,
        bboxes1: List[Tuple[int, int, int, int]],
        bboxes2: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Compute IoU matrix between two sets of bounding boxes"""
        n1 = len(bboxes1)
        n2 = len(bboxes2)
        iou_matrix = np.zeros((n1, n2))
        
        for i, box1 in enumerate(bboxes1):
            for j, box2 in enumerate(bboxes2):
                iou_matrix[i, j] = self._compute_iou(box1, box2)
        
        return iou_matrix
    
    def _compute_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Compute IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def benchmark_performance(
        self,
        images: List[np.ndarray],
        ground_truth_counts: Optional[List[int]] = None
    ) -> Dict[str, any]:
        """
        Benchmark YOLOv8 performance on a set of images.
        
        Args:
            images: List of input images
            ground_truth_counts: Optional ground truth counts for accuracy calculation
            
        Returns:
            Dictionary containing benchmark metrics
        """
        import time
        
        counts = []
        inference_times = []
        
        for image in images:
            start = time.time()
            result = self.detect(image)
            inference_times.append(time.time() - start)
            counts.append(result.count)
        
        metrics = {
            'total_images': len(images),
            'mean_count': np.mean(counts),
            'std_count': np.std(counts),
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'fps': 1.0 / np.mean(inference_times) if inference_times else 0
        }
        
        if ground_truth_counts:
            errors = [abs(pred - gt) for pred, gt in zip(counts, ground_truth_counts)]
            metrics['mae'] = np.mean(errors)
            metrics['rmse'] = np.sqrt(np.mean(np.square(errors)))
        
        return metrics
