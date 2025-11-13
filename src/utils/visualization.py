"""
Visualization Utilities
Tools for visualizing detection results, metrics, and analysis
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import seaborn as sns


class Visualizer:
    """
    Visualization tools for detection results and robustness metrics.
    
    Features:
    - Detection visualization with bounding boxes
    - Metrics radar plots
    - Comparison visualizations
    - Analysis step visualizations
    
    Args:
        save_dir: Directory to save visualizations
        dpi: DPI for saved figures
    """
    
    def __init__(
        self,
        save_dir: Optional[str] = None,
        dpi: int = 150
    ):
        self.save_dir = Path(save_dir) if save_dir else None
        self.dpi = dpi
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def visualize_detections(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        scores: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
        title: str = "Detection Results",
        save_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detection results with bounding boxes.
        
        Args:
            image: Input image (BGR format)
            bboxes: List of bounding boxes (x1, y1, x2, y2)
            scores: Optional confidence scores
            labels: Optional class labels
            title: Plot title
            save_name: Optional filename to save visualization
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        for idx, (x1, y1, x2, y2) in enumerate(bboxes):
            # Draw bounding box
            color = self._get_color(idx)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label if available
            label_text = ""
            if labels and idx < len(labels):
                label_text = labels[idx]
            if scores and idx < len(scores):
                label_text += f" {scores[idx]:.2f}"
            
            if label_text:
                # Add background for text
                (w, h), _ = cv2.getTextSize(
                    label_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1
                )
                cv2.rectangle(
                    vis_image,
                    (x1, y1 - h - 5),
                    (x1 + w, y1),
                    color,
                    -1
                )
                cv2.putText(
                    vis_image,
                    label_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        # Add count
        count_text = f"Count: {len(bboxes)}"
        cv2.putText(
            vis_image,
            count_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        if save_name and self.save_dir:
            output_path = self.save_dir / save_name
            cv2.imwrite(str(output_path), vis_image)
        
        return vis_image
    
    def plot_metrics_radar(
        self,
        metrics_dict: Dict[str, float],
        title: str = "Robustness Metrics",
        save_name: Optional[str] = None
    ):
        """
        Create radar plot for robustness metrics.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            title: Plot title
            save_name: Optional filename to save plot
        """
        categories = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        # Initialize plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], categories, size=12)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], 
                   ["0.2", "0.4", "0.6", "0.8", "1.0"], 
                   size=10)
        plt.ylim(0, 1)
        
        # Plot data
        ax.plot(angles, values, 'o-', linewidth=2, label='Metrics')
        ax.fill(angles, values, alpha=0.25)
        
        plt.title(title, size=16, y=1.08)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        if save_name and self.save_dir:
            output_path = self.save_dir / save_name
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def plot_metric_distribution(
        self,
        metric_values: List[float],
        metric_name: str,
        save_name: Optional[str] = None
    ):
        """
        Plot distribution of a metric across dataset.
        
        Args:
            metric_values: List of metric values
            metric_name: Name of the metric
            save_name: Optional filename to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        ax.hist(metric_values, bins=30, alpha=0.7, edgecolor='black')
        
        # Add statistics
        mean_val = np.mean(metric_values)
        std_val = np.std(metric_values)
        median_val = np.median(metric_values)
        
        ax.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='g', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
        
        ax.set_xlabel(metric_name, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{metric_name} Distribution (μ={mean_val:.3f}, σ={std_val:.3f})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_name and self.save_dir:
            output_path = self.save_dir / save_name
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def plot_comparison(
        self,
        method1_counts: List[int],
        method2_counts: List[int],
        method1_name: str = "Method 1",
        method2_name: str = "Method 2",
        save_name: Optional[str] = None
    ):
        """
        Create comparison plot between two methods.
        
        Args:
            method1_counts: Counts from first method
            method2_counts: Counts from second method
            method1_name: Name of first method
            method2_name: Name of second method
            save_name: Optional filename to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        ax1.scatter(method1_counts, method2_counts, alpha=0.6, s=50)
        
        # Diagonal line
        min_val = min(min(method1_counts), min(method2_counts))
        max_val = max(max(method1_counts), max(method2_counts))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        
        ax1.set_xlabel(f'{method1_name} Count', fontsize=12)
        ax1.set_ylabel(f'{method2_name} Count', fontsize=12)
        ax1.set_title('Count Comparison', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Difference histogram
        differences = np.array(method2_counts) - np.array(method1_counts)
        ax2.hist(differences, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero difference')
        ax2.axvline(np.mean(differences), color='g', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(differences):.2f}')
        
        ax2.set_xlabel('Count Difference', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Difference Distribution ({method2_name} - {method1_name})', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        if save_name and self.save_dir:
            output_path = self.save_dir / save_name
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def visualize_analysis_steps(
        self,
        intermediate_results: Dict[str, np.ndarray],
        save_name: Optional[str] = None
    ):
        """
        Visualize intermediate processing steps.
        
        Args:
            intermediate_results: Dictionary of step names and images
            save_name: Optional filename to save visualization
        """
        n_steps = len(intermediate_results)
        n_cols = min(3, n_steps)
        n_rows = (n_steps + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        for idx, (step_name, image) in enumerate(intermediate_results.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Convert BGR to RGB for display
            if len(image.shape) == 3 and image.shape[2] == 3:
                display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                display_image = image
            
            ax.imshow(display_image, cmap='gray' if len(image.shape) == 2 else None)
            ax.set_title(step_name, fontsize=12)
            ax.axis('off')
        
        # Hide extra subplots
        for idx in range(n_steps, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        if save_name and self.save_dir:
            output_path = self.save_dir / save_name
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.tight_layout()
        return fig
    
    def _get_color(self, idx: int) -> Tuple[int, int, int]:
        """Get color for visualization based on index"""
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
        return colors[idx % len(colors)]
    
    def close_all(self):
        """Close all matplotlib figures"""
        plt.close('all')
