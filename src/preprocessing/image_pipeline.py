"""
Image Preprocessing Pipeline
Handles preprocessing of UAV imagery for ornamental plant detection
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    target_size: Optional[Tuple[int, int]] = None
    normalize: bool = True
    denoise: bool = True
    enhance_contrast: bool = True
    color_correction: bool = False
    

class ImagePipeline:
    """
    Preprocessing pipeline for UAV imagery of ornamental plants.
    
    Handles:
    - Ultra-high-resolution images (4000×3000 px)
    - Variable lighting conditions
    - Color normalization for non-green species
    - Denoising and enhancement
    
    Args:
        config: PreprocessingConfig instance or dictionary
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        if config is None:
            config = PreprocessingConfig()
        elif isinstance(config, dict):
            config = PreprocessingConfig(**config)
            
        self.config = config
    
    def process(
        self,
        image: np.ndarray,
        return_intermediate: bool = False
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline to an image.
        
        Args:
            image: Input image (BGR format)
            return_intermediate: If True, return dict with intermediate results
            
        Returns:
            Processed image or dict with intermediate steps
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
        
        intermediate = {'original': image.copy()} if return_intermediate else None
        
        processed = image.copy()
        
        # Resize if specified
        if self.config.target_size:
            processed = self._resize(processed)
            if return_intermediate:
                intermediate['resized'] = processed.copy()
        
        # Denoise
        if self.config.denoise:
            processed = self._denoise(processed)
            if return_intermediate:
                intermediate['denoised'] = processed.copy()
        
        # Enhance contrast
        if self.config.enhance_contrast:
            processed = self._enhance_contrast(processed)
            if return_intermediate:
                intermediate['enhanced'] = processed.copy()
        
        # Color correction
        if self.config.color_correction:
            processed = self._color_correction(processed)
            if return_intermediate:
                intermediate['color_corrected'] = processed.copy()
        
        # Normalize
        if self.config.normalize:
            processed = self._normalize(processed)
            if return_intermediate:
                intermediate['normalized'] = processed.copy()
        
        if return_intermediate:
            intermediate['final'] = processed
            return intermediate
        
        return processed
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size"""
        return cv2.resize(
            image,
            self.config.target_size,
            interpolation=cv2.INTER_AREA
        )
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising to reduce sensor noise in UAV imagery.
        Uses Non-local Means Denoising which is effective for natural images.
        """
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Applied to LAB color space to preserve color information.
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge and convert back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _color_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color correction for consistent appearance across lighting conditions.
        Uses gray world assumption for white balance.
        """
        result = np.zeros_like(image, dtype=np.float32)
        
        for i in range(3):
            mean_val = np.mean(image[:, :, i])
            if mean_val > 0:
                result[:, :, i] = image[:, :, i] * (128.0 / mean_val)
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        return image.astype(np.float32) / 255.0
    
    def process_batch(
        self,
        images: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Process a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of processed images
        """
        return [self.process(img) for img in images]
    
    def preprocess_for_detection(
        self,
        image: np.ndarray,
        detection_size: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess image specifically for detection models.
        Returns both processed image and metadata for post-processing.
        
        Args:
            image: Input image
            detection_size: Target size for detection model
            
        Returns:
            Tuple of (processed image, metadata dict)
        """
        original_shape = image.shape[:2]
        
        # Calculate padding to maintain aspect ratio
        h, w = original_shape
        scale = min(detection_size[0] / h, detection_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to detection size
        pad_h = detection_size[0] - new_h
        pad_w = detection_size[1] - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        padded = cv2.copyMakeBorder(
            resized,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )
        
        # Apply other preprocessing
        processed = self.process(padded)
        
        metadata = {
            'original_shape': original_shape,
            'scale': scale,
            'padding': (top, bottom, left, right)
        }
        
        return processed, metadata
    
    def postprocess_detections(
        self,
        detections: List[Tuple[int, int, int, int]],
        metadata: Dict
    ) -> List[Tuple[int, int, int, int]]:
        """
        Transform detection coordinates back to original image space.
        
        Args:
            detections: List of bounding boxes (x1, y1, x2, y2)
            metadata: Metadata from preprocess_for_detection
            
        Returns:
            List of bounding boxes in original image coordinates
        """
        scale = metadata['scale']
        top, bottom, left, right = metadata['padding']
        
        transformed = []
        for x1, y1, x2, y2 in detections:
            # Remove padding
            x1 = (x1 - left) / scale
            y1 = (y1 - top) / scale
            x2 = (x2 - left) / scale
            y2 = (y2 - top) / scale
            
            transformed.append((int(x1), int(y1), int(x2), int(y2)))
        
        return transformed
