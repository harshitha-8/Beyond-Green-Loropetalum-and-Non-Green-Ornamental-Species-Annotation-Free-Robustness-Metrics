"""
Loropetalum Metrics Generation
Generate comprehensive robustness metrics for the Loropetalum dataset
"""

import numpy as np
import cv2
from typing import Dict, List, Optional
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.enhanced_loropetalum_counter import EnhancedLoropetalumCounter
from src.metrics.annotation_free_metrics import AnnotationFreeMetrics
from src.metrics.robustness_evaluator import RobustnessEvaluator
from src.utils.data_loader import DataLoader
from src.utils.visualization import Visualizer


class LoropetalumMetricsGenerator:
    """
    Generate robustness metrics for Loropetalum dataset.
    
    Produces the metrics reported in the CVPR paper:
    - RCS: 0.71 ± 0.04
    - CSC: 0.92 ± 0.05
    - SVS: 0.62 ± 0.05
    - ARI: 0.41 ± 0.09
    
    Args:
        output_dir: Directory for saving results
    """
    
    def __init__(self, output_dir: str = "./loropetalum_metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.detector = EnhancedLoropetalumCounter(
            confidence_threshold=0.5,
            nms_threshold=0.4,
            use_color_filtering=True
        )
        
        self.metrics_computer = AnnotationFreeMetrics(
            num_radial_angles=8,
            scale_factors=[0.5, 0.75, 1.0, 1.25, 1.5],
            num_augmentations=10
        )
        
        self.evaluator = RobustnessEvaluator(
            metrics_computer=self.metrics_computer,
            save_dir=str(self.output_dir)
        )
        
        self.visualizer = Visualizer(
            save_dir=str(self.output_dir / "visualizations")
        )
    
    def generate_metrics(
        self,
        data_dir: str,
        max_images: Optional[int] = None
    ) -> Dict:
        """
        Generate metrics for the Loropetalum dataset.
        
        Args:
            data_dir: Directory containing Loropetalum images
            max_images: Maximum number of images to process
            
        Returns:
            Dictionary containing all metrics and statistics
        """
        print(f"\n{'='*70}")
        print(f"LOROPETALUM METRICS GENERATION")
        print(f"{'='*70}")
        
        # Load dataset
        data_loader = DataLoader(data_dir)
        n_images = min(len(data_loader), max_images) if max_images else len(data_loader)
        
        print(f"\nDataset: {data_dir}")
        print(f"Processing: {n_images} images")
        
        # Process images
        all_metrics = {
            'RCS': [],
            'CSC': [],
            'SVS': [],
            'ARI': []
        }
        
        for i in range(n_images):
            image = data_loader.load_image(i)
            image_name = data_loader.get_image_name(i)
            
            print(f"\nProcessing [{i+1}/{n_images}]: {image_name}")
            
            # Define detector and segmentation functions
            detector_func = lambda img: self.detector.detect(img).count
            segmentation_func = lambda img: self.detector.detect(img, return_masks=True).segmentation_masks
            
            # Compute metrics
            result = self.evaluator.evaluate_image(
                image,
                detector_func,
                segmentation_func,
                image_name
            )
            
            # Collect metrics
            for metric_name in all_metrics.keys():
                all_metrics[metric_name].append(result['metrics'][metric_name])
            
            print(f"  RCS: {result['metrics']['RCS']:.3f}")
            print(f"  CSC: {result['metrics']['CSC']:.3f}")
            print(f"  SVS: {result['metrics']['SVS']:.3f}")
            print(f"  ARI: {result['metrics']['ARI']:.3f}")
        
        # Compute statistics
        statistics = {}
        for metric_name, values in all_metrics.items():
            statistics[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        # Generate visualizations
        self._create_visualizations(all_metrics)
        
        # Generate report
        report = self._generate_report(statistics, n_images)
        print(f"\n{report}")
        
        # Save results
        results = {
            'num_images': n_images,
            'statistics': statistics,
            'individual_values': all_metrics
        }
        
        results_path = self.output_dir / "metrics_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        report_path = self.output_dir / "metrics_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nResults saved to: {self.output_dir}")
        
        return results
    
    def _create_visualizations(self, all_metrics: Dict[str, List[float]]):
        """Create visualization plots for metrics"""
        # Plot distributions for each metric
        for metric_name, values in all_metrics.items():
            self.visualizer.plot_metric_distribution(
                values,
                metric_name,
                save_name=f"{metric_name}_distribution.png"
            )
        
        # Create radar plot with mean values
        mean_metrics = {
            name: np.mean(values)
            for name, values in all_metrics.items()
        }
        self.visualizer.plot_metrics_radar(
            mean_metrics,
            title="Mean Robustness Metrics - Loropetalum Dataset",
            save_name="mean_metrics_radar.png"
        )
    
    def _generate_report(self, statistics: Dict, n_images: int) -> str:
        """Generate text report of metrics"""
        report = []
        report.append("=" * 70)
        report.append("LOROPETALUM ROBUSTNESS METRICS REPORT")
        report.append("=" * 70)
        report.append(f"\nDataset: Loropetalum chinense UAV imagery")
        report.append(f"Total Images: {n_images}")
        
        report.append("\n" + "-" * 70)
        report.append("ANNOTATION-FREE ROBUSTNESS METRICS")
        report.append("-" * 70)
        
        for metric_name in ['RCS', 'CSC', 'SVS', 'ARI']:
            stats = statistics[metric_name]
            report.append(f"\n{metric_name}:")
            report.append(f"  Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")
            report.append(f"  Median:     {stats['median']:.3f}")
            report.append(f"  Range:      [{stats['min']:.3f}, {stats['max']:.3f}]")
        
        report.append("\n" + "-" * 70)
        report.append("INTERPRETATION")
        report.append("-" * 70)
        report.append("\nRCS (Radial Counting Stability):")
        report.append("  Measures spatial prediction consistency under rotations")
        report.append("  Higher values indicate more stable counting across orientations")
        
        report.append("\nCSC (Cross-Scale Consistency):")
        report.append("  Evaluates robustness across different UAV flight altitudes")
        report.append("  Higher values indicate consistent detection at multiple scales")
        
        report.append("\nSVS (Semantic-Visual Stability):")
        report.append("  Quantifies segmentation coherence without ground truth")
        report.append("  Higher values indicate better alignment with visual features")
        
        report.append("\nARI (Adaptive Repeatability Index):")
        report.append("  Assesses stochastic consistency under augmentations")
        report.append("  Higher values indicate more repeatable predictions")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Loropetalum Metrics Generation"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing Loropetalum images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./loropetalum_metrics',
        help='Output directory'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to process'
    )
    
    args = parser.parse_args()
    
    generator = LoropetalumMetricsGenerator(output_dir=args.output)
    result = generator.generate_metrics(
        args.data_dir,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()
