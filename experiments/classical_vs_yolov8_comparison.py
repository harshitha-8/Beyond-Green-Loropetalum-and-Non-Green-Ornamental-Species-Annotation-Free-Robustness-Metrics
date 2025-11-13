"""
Classical vs YOLOv8 Comparison
Compares classical computer vision methods with YOLOv8 deep learning approach
"""

import numpy as np
import cv2
from typing import Dict, List, Optional
from pathlib import Path
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.enhanced_loropetalum_counter import EnhancedLoropetalumCounter
from src.models.yolov8_comparison import YOLOv8Comparator
from src.utils.data_loader import DataLoader
from src.utils.visualization import Visualizer


class ClassicalVsYOLOv8Comparison:
    """
    Comparison framework for classical and deep learning approaches.
    
    Evaluates:
    - Counting accuracy
    - Processing speed
    - Detection quality (IoU)
    - Robustness to perturbations
    
    Args:
        yolov8_model_path: Optional path to YOLOv8 model weights
        output_dir: Directory for saving results
    """
    
    def __init__(
        self,
        yolov8_model_path: Optional[str] = None,
        output_dir: str = "./comparison_results"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize methods
        self.classical = EnhancedLoropetalumCounter(
            confidence_threshold=0.5,
            nms_threshold=0.4,
            use_color_filtering=True
        )
        
        self.yolov8 = YOLOv8Comparator(
            model_path=yolov8_model_path,
            confidence_threshold=0.5,
            iou_threshold=0.45
        )
        
        self.visualizer = Visualizer(save_dir=str(self.output_dir))
        
        self.results = []
    
    def compare_single_image(
        self,
        image: np.ndarray,
        image_name: str = "image"
    ) -> Dict:
        """
        Compare both methods on a single image.
        
        Args:
            image: Input image (BGR format)
            image_name: Name for tracking
            
        Returns:
            Dictionary containing comparison results
        """
        print(f"\nComparing methods on: {image_name}")
        
        # Classical method
        start = time.time()
        classical_result = self.classical.detect(image, return_masks=True)
        classical_time = time.time() - start
        
        # YOLOv8 method
        start = time.time()
        yolov8_result = self.yolov8.detect(image)
        yolov8_time = time.time() - start
        
        # Compute comparison metrics
        comparison = {
            'image_name': image_name,
            'classical': {
                'count': classical_result.count,
                'time': classical_time,
                'fps': 1.0 / classical_time if classical_time > 0 else 0,
                'bboxes': classical_result.bboxes,
                'scores': classical_result.confidence_scores
            },
            'yolov8': {
                'count': yolov8_result.count,
                'time': yolov8_time,
                'fps': 1.0 / yolov8_time if yolov8_time > 0 else 0,
                'bboxes': yolov8_result.bboxes,
                'scores': yolov8_result.confidence_scores
            }
        }
        
        # IoU-based matching
        if classical_result.bboxes and yolov8_result.bboxes:
            iou_matrix = self._compute_iou_matrix(
                classical_result.bboxes,
                yolov8_result.bboxes
            )
            comparison['mean_iou'] = float(np.mean(iou_matrix[iou_matrix > 0.1]))
            comparison['matched_detections'] = int(np.sum(np.max(iou_matrix, axis=1) > 0.5))
        else:
            comparison['mean_iou'] = 0.0
            comparison['matched_detections'] = 0
        
        comparison['count_difference'] = abs(
            classical_result.count - yolov8_result.count
        )
        comparison['speedup'] = classical_time / yolov8_time if yolov8_time > 0 else 0
        
        self.results.append(comparison)
        
        print(f"  Classical: {classical_result.count} plants, {classical_time:.3f}s")
        print(f"  YOLOv8:    {yolov8_result.count} plants, {yolov8_time:.3f}s")
        print(f"  Count difference: {comparison['count_difference']}")
        print(f"  Mean IoU: {comparison['mean_iou']:.3f}")
        
        return comparison
    
    def compare_dataset(
        self,
        data_dir: str,
        max_images: Optional[int] = None,
        visualize_samples: int = 5
    ) -> Dict:
        """
        Compare methods on an entire dataset.
        
        Args:
            data_dir: Directory containing images
            max_images: Maximum number of images to process
            visualize_samples: Number of samples to visualize
            
        Returns:
            Dictionary containing aggregated comparison results
        """
        print(f"\n{'='*70}")
        print(f"CLASSICAL VS YOLOV8 COMPARISON")
        print(f"{'='*70}")
        
        # Load dataset
        data_loader = DataLoader(data_dir)
        n_images = min(len(data_loader), max_images) if max_images else len(data_loader)
        
        print(f"\nDataset: {data_dir}")
        print(f"Processing: {n_images} images")
        
        # Process images
        for i in range(n_images):
            image = data_loader.load_image(i)
            image_name = data_loader.get_image_name(i)
            
            comparison = self.compare_single_image(image, image_name)
            
            # Visualize samples
            if i < visualize_samples:
                self._visualize_comparison(image, comparison, i)
        
        # Aggregate results
        aggregated = self._aggregate_results()
        
        # Generate report
        report = self._generate_report(aggregated)
        print(f"\n{report}")
        
        # Save report
        report_path = self.output_dir / "comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Create comparison plots
        self._create_comparison_plots()
        
        print(f"\nResults saved to: {self.output_dir}")
        
        return aggregated
    
    def _visualize_comparison(
        self,
        image: np.ndarray,
        comparison: Dict,
        idx: int
    ):
        """Create side-by-side visualization of both methods"""
        # Classical visualization
        classical_vis = self.visualizer.visualize_detections(
            image.copy(),
            comparison['classical']['bboxes'],
            comparison['classical']['scores'],
            save_name=f"sample_{idx:03d}_classical.jpg"
        )
        
        # YOLOv8 visualization
        yolov8_vis = self.visualizer.visualize_detections(
            image.copy(),
            comparison['yolov8']['bboxes'],
            comparison['yolov8']['scores'],
            save_name=f"sample_{idx:03d}_yolov8.jpg"
        )
    
    def _aggregate_results(self) -> Dict:
        """Aggregate comparison results across dataset"""
        if not self.results:
            return {}
        
        classical_counts = [r['classical']['count'] for r in self.results]
        yolov8_counts = [r['yolov8']['count'] for r in self.results]
        classical_times = [r['classical']['time'] for r in self.results]
        yolov8_times = [r['yolov8']['time'] for r in self.results]
        count_diffs = [r['count_difference'] for r in self.results]
        mean_ious = [r['mean_iou'] for r in self.results]
        
        return {
            'num_images': len(self.results),
            'classical': {
                'mean_count': float(np.mean(classical_counts)),
                'std_count': float(np.std(classical_counts)),
                'mean_time': float(np.mean(classical_times)),
                'mean_fps': float(np.mean([r['classical']['fps'] for r in self.results]))
            },
            'yolov8': {
                'mean_count': float(np.mean(yolov8_counts)),
                'std_count': float(np.std(yolov8_counts)),
                'mean_time': float(np.mean(yolov8_times)),
                'mean_fps': float(np.mean([r['yolov8']['fps'] for r in self.results]))
            },
            'comparison': {
                'mean_count_difference': float(np.mean(count_diffs)),
                'mean_iou': float(np.mean(mean_ious)),
                'speedup': float(np.mean(classical_times) / np.mean(yolov8_times)) if np.mean(yolov8_times) > 0 else 0
            }
        }
    
    def _generate_report(self, aggregated: Dict) -> str:
        """Generate text report of comparison results"""
        report = []
        report.append("=" * 70)
        report.append("CLASSICAL VS YOLOV8 COMPARISON REPORT")
        report.append("=" * 70)
        report.append(f"\nTotal Images: {aggregated['num_images']}")
        
        report.append("\n" + "-" * 70)
        report.append("CLASSICAL METHOD")
        report.append("-" * 70)
        classical = aggregated['classical']
        report.append(f"Mean Count:  {classical['mean_count']:.2f} ± {classical['std_count']:.2f}")
        report.append(f"Mean Time:   {classical['mean_time']:.3f}s")
        report.append(f"Mean FPS:    {classical['mean_fps']:.2f}")
        
        report.append("\n" + "-" * 70)
        report.append("YOLOV8 METHOD")
        report.append("-" * 70)
        yolov8 = aggregated['yolov8']
        report.append(f"Mean Count:  {yolov8['mean_count']:.2f} ± {yolov8['std_count']:.2f}")
        report.append(f"Mean Time:   {yolov8['mean_time']:.3f}s")
        report.append(f"Mean FPS:    {yolov8['mean_fps']:.2f}")
        
        report.append("\n" + "-" * 70)
        report.append("COMPARISON")
        report.append("-" * 70)
        comp = aggregated['comparison']
        report.append(f"Mean Count Difference: {comp['mean_count_difference']:.2f}")
        report.append(f"Mean IoU:              {comp['mean_iou']:.3f}")
        report.append(f"Speed Ratio:           {comp['speedup']:.2f}x")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def _create_comparison_plots(self):
        """Create comparison visualization plots"""
        if not self.results:
            return
        
        classical_counts = [r['classical']['count'] for r in self.results]
        yolov8_counts = [r['yolov8']['count'] for r in self.results]
        
        self.visualizer.plot_comparison(
            classical_counts,
            yolov8_counts,
            "Classical",
            "YOLOv8",
            save_name="count_comparison.png"
        )
    
    def _compute_iou_matrix(
        self,
        bboxes1: List,
        bboxes2: List
    ) -> np.ndarray:
        """Compute IoU matrix between two sets of bounding boxes"""
        n1 = len(bboxes1)
        n2 = len(bboxes2)
        iou_matrix = np.zeros((n1, n2))
        
        for i, box1 in enumerate(bboxes1):
            for j, box2 in enumerate(bboxes2):
                iou_matrix[i, j] = self._compute_iou(box1, box2)
        
        return iou_matrix
    
    def _compute_iou(self, box1, box2) -> float:
        """Compute IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Classical vs YOLOv8 Comparison"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing images'
    )
    parser.add_argument(
        '--yolov8-model',
        type=str,
        default=None,
        help='Path to YOLOv8 model weights'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./comparison_results',
        help='Output directory'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to process'
    )
    
    args = parser.parse_args()
    
    comparator = ClassicalVsYOLOv8Comparison(
        yolov8_model_path=args.yolov8_model,
        output_dir=args.output
    )
    
    result = comparator.compare_dataset(
        args.data_dir,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()
