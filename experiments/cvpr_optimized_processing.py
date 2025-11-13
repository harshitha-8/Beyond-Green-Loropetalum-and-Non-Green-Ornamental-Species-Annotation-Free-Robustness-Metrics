"""
CVPR Optimized Processing Pipeline
Implements the optimized processing pipeline described in the CVPR submission
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.enhanced_loropetalum_counter import EnhancedLoropetalumCounter
from src.metrics.annotation_free_metrics import AnnotationFreeMetrics
from src.metrics.robustness_evaluator import RobustnessEvaluator
from src.preprocessing.image_pipeline import ImagePipeline, PreprocessingConfig
from src.utils.data_loader import DataLoader
from src.utils.visualization import Visualizer


class CVPROptimizedProcessor:
    """
    Optimized processing pipeline for CVPR paper.
    
    Implements the complete pipeline:
    1. Image preprocessing
    2. Plant detection and counting
    3. Annotation-free robustness metric computation
    4. Visualization and result generation
    
    Args:
        output_dir: Directory for saving results
        config: Optional processing configuration
    """
    
    def __init__(
        self,
        output_dir: str = "./results",
        config: Optional[Dict] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.preprocessor = ImagePipeline(
            PreprocessingConfig(
                denoise=True,
                enhance_contrast=True,
                normalize=False
            )
        )
        
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
            save_dir=str(self.output_dir / "metrics")
        )
        
        self.visualizer = Visualizer(
            save_dir=str(self.output_dir / "visualizations")
        )
    
    def process_single_image(
        self,
        image_path: str,
        visualize: bool = True
    ) -> Dict:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to input image
            visualize: Whether to generate visualizations
            
        Returns:
            Dictionary containing all results
        """
        print(f"\nProcessing: {image_path}")
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Preprocess
        preprocessed = self.preprocessor.process(image)
        
        # Detect plants
        detection_result = self.detector.detect(preprocessed, return_masks=True)
        
        # Compute robustness metrics
        detector_func = lambda img: self.detector.detect(img).count
        segmentation_func = lambda img: self.detector.detect(img, return_masks=True).segmentation_masks
        
        eval_result = self.evaluator.evaluate_image(
            preprocessed,
            detector_func,
            segmentation_func,
            Path(image_path).name
        )
        
        processing_time = time.time() - start_time
        
        # Generate visualizations
        if visualize:
            vis_image = self.visualizer.visualize_detections(
                image,
                detection_result.bboxes,
                detection_result.confidence_scores,
                save_name=f"{Path(image_path).stem}_detections.jpg"
            )
            
            self.visualizer.plot_metrics_radar(
                eval_result['metrics'],
                title=f"Robustness Metrics - {Path(image_path).name}",
                save_name=f"{Path(image_path).stem}_metrics_radar.png"
            )
        
        result = {
            'image_path': image_path,
            'count': detection_result.count,
            'metrics': eval_result['metrics'],
            'overall_score': eval_result['overall_score'],
            'processing_time': processing_time
        }
        
        print(f"  Count: {result['count']}")
        print(f"  RCS: {result['metrics']['RCS']:.3f}")
        print(f"  CSC: {result['metrics']['CSC']:.3f}")
        print(f"  SVS: {result['metrics']['SVS']:.3f}")
        print(f"  ARI: {result['metrics']['ARI']:.3f}")
        print(f"  Overall Score: {result['overall_score']:.3f}")
        print(f"  Processing Time: {processing_time:.2f}s")
        
        return result
    
    def process_dataset(
        self,
        data_dir: str,
        max_images: Optional[int] = None,
        visualize_samples: int = 5
    ) -> Dict:
        """
        Process an entire dataset.
        
        Args:
            data_dir: Directory containing images
            max_images: Maximum number of images to process
            visualize_samples: Number of samples to visualize
            
        Returns:
            Dictionary containing aggregated results
        """
        print(f"\n{'='*70}")
        print(f"CVPR OPTIMIZED PROCESSING PIPELINE")
        print(f"{'='*70}")
        
        # Load dataset
        data_loader = DataLoader(data_dir)
        n_images = min(len(data_loader), max_images) if max_images else len(data_loader)
        
        print(f"\nDataset: {data_dir}")
        print(f"Total images: {len(data_loader)}")
        print(f"Processing: {n_images} images")
        
        # Process images
        results = []
        for i in range(n_images):
            image = data_loader.load_image(i)
            image_name = data_loader.get_image_name(i)
            
            # Detect plants
            detection_result = self.detector.detect(image, return_masks=True)
            
            # Compute metrics
            detector_func = lambda img: self.detector.detect(img).count
            segmentation_func = lambda img: self.detector.detect(img, return_masks=True).segmentation_masks
            
            eval_result = self.evaluator.evaluate_image(
                image,
                detector_func,
                segmentation_func,
                image_name
            )
            
            results.append({
                'image_name': image_name,
                'count': detection_result.count,
                **eval_result
            })
            
            # Visualize samples
            if i < visualize_samples:
                self.visualizer.visualize_detections(
                    image,
                    detection_result.bboxes,
                    detection_result.confidence_scores,
                    save_name=f"sample_{i:03d}_detections.jpg"
                )
        
        # Generate final report
        report = self.evaluator.generate_report()
        print(f"\n{report}")
        
        # Save report
        report_path = self.output_dir / "processing_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nResults saved to: {self.output_dir}")
        
        return {
            'results': results,
            'summary': self.evaluator._aggregate_results(results)
        }


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CVPR Optimized Processing Pipeline"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input image or directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./results',
        help='Output directory'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to process'
    )
    
    args = parser.parse_args()
    
    processor = CVPROptimizedProcessor(output_dir=args.output)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single image
        result = processor.process_single_image(str(input_path))
    elif input_path.is_dir():
        # Process dataset
        result = processor.process_dataset(
            str(input_path),
            max_images=args.max_images
        )
    else:
        raise ValueError(f"Invalid input path: {args.input}")


if __name__ == "__main__":
    main()
