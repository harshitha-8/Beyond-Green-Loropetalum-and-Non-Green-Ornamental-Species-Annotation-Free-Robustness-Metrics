"""
Strategy Evaluation
Evaluate different detection strategies and their robustness
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Callable
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.enhanced_loropetalum_counter import EnhancedLoropetalumCounter
from src.metrics.annotation_free_metrics import AnnotationFreeMetrics
from src.preprocessing.image_pipeline import ImagePipeline, PreprocessingConfig
from src.utils.data_loader import DataLoader


class StrategyEvaluator:
    """
    Evaluate different detection strategies using annotation-free metrics.
    
    Strategies evaluated:
    1. Color-based filtering (Loropetalum-specific)
    2. Edge-based detection
    3. Hybrid approach
    4. Different preprocessing configurations
    
    Args:
        output_dir: Directory for saving results
    """
    
    def __init__(self, output_dir: str = "./strategy_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_computer = AnnotationFreeMetrics()
        self.strategies = self._define_strategies()
    
    def _define_strategies(self) -> Dict[str, Callable]:
        """Define different detection strategies to evaluate"""
        strategies = {}
        
        # Strategy 1: Color-based (default)
        def color_based_strategy(image):
            detector = EnhancedLoropetalumCounter(
                confidence_threshold=0.5,
                use_color_filtering=True
            )
            return detector.detect(image).count
        
        strategies['color_based'] = color_based_strategy
        
        # Strategy 2: Without color filtering
        def no_color_strategy(image):
            detector = EnhancedLoropetalumCounter(
                confidence_threshold=0.5,
                use_color_filtering=False
            )
            return detector.detect(image).count
        
        strategies['no_color'] = no_color_strategy
        
        # Strategy 3: High confidence threshold
        def high_confidence_strategy(image):
            detector = EnhancedLoropetalumCounter(
                confidence_threshold=0.7,
                use_color_filtering=True
            )
            return detector.detect(image).count
        
        strategies['high_confidence'] = high_confidence_strategy
        
        # Strategy 4: Low confidence threshold
        def low_confidence_strategy(image):
            detector = EnhancedLoropetalumCounter(
                confidence_threshold=0.3,
                use_color_filtering=True
            )
            return detector.detect(image).count
        
        strategies['low_confidence'] = low_confidence_strategy
        
        # Strategy 5: With preprocessing
        def preprocessed_strategy(image):
            pipeline = ImagePipeline(
                PreprocessingConfig(
                    denoise=True,
                    enhance_contrast=True
                )
            )
            processed = pipeline.process(image)
            detector = EnhancedLoropetalumCounter(use_color_filtering=True)
            return detector.detect(processed).count
        
        strategies['with_preprocessing'] = preprocessed_strategy
        
        return strategies
    
    def evaluate_strategies(
        self,
        data_dir: str,
        max_images: Optional[int] = None
    ) -> Dict:
        """
        Evaluate all strategies on a dataset.
        
        Args:
            data_dir: Directory containing images
            max_images: Maximum number of images to process
            
        Returns:
            Dictionary containing evaluation results for all strategies
        """
        print(f"\n{'='*70}")
        print(f"STRATEGY EVALUATION")
        print(f"{'='*70}")
        
        # Load dataset
        data_loader = DataLoader(data_dir)
        n_images = min(len(data_loader), max_images) if max_images else len(data_loader)
        
        print(f"\nDataset: {data_dir}")
        print(f"Strategies: {len(self.strategies)}")
        print(f"Processing: {n_images} images")
        
        # Evaluate each strategy
        results = {}
        
        for strategy_name, strategy_func in self.strategies.items():
            print(f"\n{'-'*70}")
            print(f"Evaluating Strategy: {strategy_name}")
            print(f"{'-'*70}")
            
            strategy_metrics = {
                'RCS': [],
                'CSC': [],
                'SVS': [],
                'ARI': []
            }
            
            for i in range(n_images):
                image = data_loader.load_image(i)
                
                # Compute metrics for this strategy
                metrics = self.metrics_computer.compute_all_metrics(
                    image,
                    strategy_func,
                    None  # No segmentation for now
                )
                
                strategy_metrics['RCS'].append(metrics.rcs)
                strategy_metrics['CSC'].append(metrics.csc)
                strategy_metrics['SVS'].append(metrics.svs)
                strategy_metrics['ARI'].append(metrics.ari)
            
            # Compute statistics
            strategy_stats = {}
            for metric_name, values in strategy_metrics.items():
                strategy_stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values))
                }
            
            results[strategy_name] = strategy_stats
            
            # Print summary
            print(f"  RCS: {strategy_stats['RCS']['mean']:.3f} ± {strategy_stats['RCS']['std']:.3f}")
            print(f"  CSC: {strategy_stats['CSC']['mean']:.3f} ± {strategy_stats['CSC']['std']:.3f}")
            print(f"  SVS: {strategy_stats['SVS']['mean']:.3f} ± {strategy_stats['SVS']['std']:.3f}")
            print(f"  ARI: {strategy_stats['ARI']['mean']:.3f} ± {strategy_stats['ARI']['std']:.3f}")
        
        # Generate comparison report
        report = self._generate_comparison_report(results)
        print(f"\n{report}")
        
        # Save results
        output = {
            'num_images': n_images,
            'strategies': results
        }
        
        results_path = self.output_dir / "strategy_evaluation.json"
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        report_path = self.output_dir / "strategy_comparison.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nResults saved to: {self.output_dir}")
        
        return output
    
    def _generate_comparison_report(self, results: Dict) -> str:
        """Generate comparison report for all strategies"""
        report = []
        report.append("=" * 70)
        report.append("STRATEGY COMPARISON REPORT")
        report.append("=" * 70)
        
        # For each metric, compare strategies
        for metric_name in ['RCS', 'CSC', 'SVS', 'ARI']:
            report.append(f"\n{'-'*70}")
            report.append(f"{metric_name} COMPARISON")
            report.append(f"{'-'*70}")
            
            # Sort strategies by mean metric value
            sorted_strategies = sorted(
                results.items(),
                key=lambda x: x[1][metric_name]['mean'],
                reverse=True
            )
            
            for rank, (strategy_name, stats) in enumerate(sorted_strategies, 1):
                metric_stats = stats[metric_name]
                report.append(
                    f"{rank}. {strategy_name:25s} "
                    f"{metric_stats['mean']:.3f} ± {metric_stats['std']:.3f}"
                )
        
        # Overall ranking (by average of all metrics)
        report.append(f"\n{'='*70}")
        report.append("OVERALL RANKING (Average of All Metrics)")
        report.append(f"{'='*70}")
        
        overall_scores = {}
        for strategy_name, stats in results.items():
            avg_score = np.mean([
                stats['RCS']['mean'],
                stats['CSC']['mean'],
                stats['SVS']['mean'],
                stats['ARI']['mean']
            ])
            overall_scores[strategy_name] = avg_score
        
        sorted_overall = sorted(
            overall_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for rank, (strategy_name, score) in enumerate(sorted_overall, 1):
            report.append(f"{rank}. {strategy_name:25s} {score:.3f}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Strategy Evaluation"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./strategy_evaluation',
        help='Output directory'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to process'
    )
    
    args = parser.parse_args()
    
    evaluator = StrategyEvaluator(output_dir=args.output)
    result = evaluator.evaluate_strategies(
        args.data_dir,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()
