"""
Robustness Evaluator
High-level interface for evaluating model robustness using annotation-free metrics
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Callable
from pathlib import Path
import json
from dataclasses import asdict

from .annotation_free_metrics import AnnotationFreeMetrics, RobustnessMetrics


class RobustnessEvaluator:
    """
    High-level evaluator for assessing model robustness on image datasets.
    
    Features:
    - Batch evaluation on multiple images
    - Statistical aggregation of metrics
    - Result persistence and visualization
    - Correlation with deployment success
    
    Args:
        metrics_computer: AnnotationFreeMetrics instance
        save_dir: Directory to save evaluation results
    """
    
    def __init__(
        self,
        metrics_computer: Optional[AnnotationFreeMetrics] = None,
        save_dir: Optional[str] = None
    ):
        self.metrics_computer = metrics_computer or AnnotationFreeMetrics()
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_cache = []
    
    def evaluate_image(
        self,
        image: np.ndarray,
        detector_func: Callable,
        segmentation_func: Optional[Callable] = None,
        image_name: Optional[str] = None
    ) -> Dict:
        """
        Evaluate robustness metrics for a single image.
        
        Args:
            image: Input image (BGR format)
            detector_func: Function that takes image and returns count
            segmentation_func: Optional function for segmentation
            image_name: Optional name for tracking
            
        Returns:
            Dictionary containing metrics and metadata
        """
        # Compute metrics
        metrics = self.metrics_computer.compute_all_metrics(
            image,
            detector_func,
            segmentation_func
        )
        
        # Package results
        result = {
            'image_name': image_name or 'unknown',
            'image_shape': image.shape,
            'metrics': metrics.to_dict(),
            'overall_score': metrics.overall_score()
        }
        
        # Cache results
        self.results_cache.append(result)
        
        return result
    
    def evaluate_dataset(
        self,
        images: List[np.ndarray],
        detector_func: Callable,
        segmentation_func: Optional[Callable] = None,
        image_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate robustness metrics for a dataset of images.
        
        Args:
            images: List of input images
            detector_func: Detection function
            segmentation_func: Optional segmentation function
            image_names: Optional list of image names
            
        Returns:
            Dictionary containing aggregated metrics and statistics
        """
        if image_names is None:
            image_names = [f"image_{i:04d}" for i in range(len(images))]
        
        results = []
        for image, name in zip(images, image_names):
            result = self.evaluate_image(
                image,
                detector_func,
                segmentation_func,
                name
            )
            results.append(result)
        
        # Aggregate statistics
        aggregated = self._aggregate_results(results)
        
        # Save if directory specified
        if self.save_dir:
            self._save_results(aggregated)
        
        return aggregated
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate metrics across multiple images"""
        if not results:
            return {}
        
        # Extract metric arrays
        rcs_values = [r['metrics']['RCS'] for r in results]
        csc_values = [r['metrics']['CSC'] for r in results]
        svs_values = [r['metrics']['SVS'] for r in results]
        ari_values = [r['metrics']['ARI'] for r in results]
        overall_scores = [r['overall_score'] for r in results]
        
        aggregated = {
            'num_images': len(results),
            'individual_results': results,
            'statistics': {
                'RCS': {
                    'mean': float(np.mean(rcs_values)),
                    'std': float(np.std(rcs_values)),
                    'min': float(np.min(rcs_values)),
                    'max': float(np.max(rcs_values)),
                    'median': float(np.median(rcs_values))
                },
                'CSC': {
                    'mean': float(np.mean(csc_values)),
                    'std': float(np.std(csc_values)),
                    'min': float(np.min(csc_values)),
                    'max': float(np.max(csc_values)),
                    'median': float(np.median(csc_values))
                },
                'SVS': {
                    'mean': float(np.mean(svs_values)),
                    'std': float(np.std(svs_values)),
                    'min': float(np.min(svs_values)),
                    'max': float(np.max(svs_values)),
                    'median': float(np.median(svs_values))
                },
                'ARI': {
                    'mean': float(np.mean(ari_values)),
                    'std': float(np.std(ari_values)),
                    'min': float(np.min(ari_values)),
                    'max': float(np.max(ari_values)),
                    'median': float(np.median(ari_values))
                },
                'overall_score': {
                    'mean': float(np.mean(overall_scores)),
                    'std': float(np.std(overall_scores)),
                    'min': float(np.min(overall_scores)),
                    'max': float(np.max(overall_scores)),
                    'median': float(np.median(overall_scores))
                }
            }
        }
        
        return aggregated
    
    def _save_results(self, results: Dict):
        """Save evaluation results to disk"""
        output_path = self.save_dir / 'evaluation_results.json'
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def compute_correlation_with_ground_truth(
        self,
        ground_truth_scores: List[float]
    ) -> Dict[str, float]:
        """
        Compute correlation between annotation-free metrics and ground truth.
        
        Args:
            ground_truth_scores: List of ground truth performance scores
            
        Returns:
            Dictionary containing correlation coefficients
        """
        from scipy.stats import spearmanr, pearsonr
        
        if len(self.results_cache) != len(ground_truth_scores):
            raise ValueError("Number of results must match ground truth scores")
        
        # Extract metric values
        rcs_values = [r['metrics']['RCS'] for r in self.results_cache]
        csc_values = [r['metrics']['CSC'] for r in self.results_cache]
        svs_values = [r['metrics']['SVS'] for r in self.results_cache]
        ari_values = [r['metrics']['ARI'] for r in self.results_cache]
        overall_scores = [r['overall_score'] for r in self.results_cache]
        
        correlations = {}
        
        for name, values in [
            ('RCS', rcs_values),
            ('CSC', csc_values),
            ('SVS', svs_values),
            ('ARI', ari_values),
            ('overall', overall_scores)
        ]:
            spearman_corr, spearman_p = spearmanr(values, ground_truth_scores)
            pearson_corr, pearson_p = pearsonr(values, ground_truth_scores)
            
            correlations[name] = {
                'spearman_rho': float(spearman_corr),
                'spearman_p_value': float(spearman_p),
                'pearson_r': float(pearson_corr),
                'pearson_p_value': float(pearson_p)
            }
        
        return correlations
    
    def generate_report(self) -> str:
        """
        Generate a text report summarizing evaluation results.
        
        Returns:
            Formatted text report
        """
        if not self.results_cache:
            return "No evaluation results available."
        
        aggregated = self._aggregate_results(self.results_cache)
        stats = aggregated['statistics']
        
        report = []
        report.append("=" * 70)
        report.append("ROBUSTNESS EVALUATION REPORT")
        report.append("=" * 70)
        report.append(f"\nTotal Images Evaluated: {aggregated['num_images']}")
        report.append("\n" + "-" * 70)
        report.append("METRIC STATISTICS")
        report.append("-" * 70)
        
        for metric_name in ['RCS', 'CSC', 'SVS', 'ARI']:
            metric_stats = stats[metric_name]
            report.append(f"\n{metric_name} (Higher is Better):")
            report.append(f"  Mean:   {metric_stats['mean']:.3f} ± {metric_stats['std']:.3f}")
            report.append(f"  Median: {metric_stats['median']:.3f}")
            report.append(f"  Range:  [{metric_stats['min']:.3f}, {metric_stats['max']:.3f}]")
        
        overall = stats['overall_score']
        report.append(f"\nOverall Robustness Score:")
        report.append(f"  Mean:   {overall['mean']:.3f} ± {overall['std']:.3f}")
        report.append(f"  Median: {overall['median']:.3f}")
        report.append(f"  Range:  [{overall['min']:.3f}, {overall['max']:.3f}]")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def clear_cache(self):
        """Clear cached results"""
        self.results_cache = []
