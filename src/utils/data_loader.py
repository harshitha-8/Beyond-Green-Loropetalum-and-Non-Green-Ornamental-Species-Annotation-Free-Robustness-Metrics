"""
Data Loading Utilities
Tools for loading and managing UAV imagery datasets
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Generator
from pathlib import Path
import json


class DataLoader:
    """
    Data loader for UAV imagery datasets.
    
    Features:
    - Load images from directory
    - Support for multiple image formats
    - Batch loading with memory management
    - Metadata management
    - Dataset splitting
    
    Args:
        data_dir: Path to data directory
        image_extensions: Tuple of valid image extensions
        cache_size: Maximum number of images to cache in memory
    """
    
    def __init__(
        self,
        data_dir: str,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.tiff', '.tif'),
        cache_size: int = 100
    ):
        self.data_dir = Path(data_dir)
        self.image_extensions = image_extensions
        self.cache_size = cache_size
        
        self.image_paths = []
        self.metadata = {}
        self._cache = {}
        
        if self.data_dir.exists():
            self._discover_images()
    
    def _discover_images(self):
        """Discover all images in data directory"""
        self.image_paths = []
        
        for ext in self.image_extensions:
            self.image_paths.extend(self.data_dir.glob(f'**/*{ext}'))
            self.image_paths.extend(self.data_dir.glob(f'**/*{ext.upper()}'))
        
        self.image_paths = sorted(self.image_paths)
        print(f"Discovered {len(self.image_paths)} images in {self.data_dir}")
    
    def load_image(
        self,
        idx: int,
        use_cache: bool = True
    ) -> Optional[np.ndarray]:
        """
        Load a single image by index.
        
        Args:
            idx: Image index
            use_cache: Whether to use cache
            
        Returns:
            Image as numpy array or None if loading fails
        """
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range [0, {len(self.image_paths)})")
        
        image_path = self.image_paths[idx]
        
        # Check cache
        if use_cache and str(image_path) in self._cache:
            return self._cache[str(image_path)]
        
        # Load image
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"Warning: Failed to load {image_path}")
            return None
        
        # Update cache
        if use_cache:
            self._update_cache(str(image_path), image)
        
        return image
    
    def load_batch(
        self,
        indices: List[int],
        use_cache: bool = True
    ) -> List[np.ndarray]:
        """
        Load multiple images by indices.
        
        Args:
            indices: List of image indices
            use_cache: Whether to use cache
            
        Returns:
            List of images
        """
        images = []
        for idx in indices:
            image = self.load_image(idx, use_cache)
            if image is not None:
                images.append(image)
        return images
    
    def load_all(
        self,
        use_cache: bool = False
    ) -> List[np.ndarray]:
        """
        Load all images in dataset.
        Warning: May consume large amounts of memory!
        
        Args:
            use_cache: Whether to use cache
            
        Returns:
            List of all images
        """
        return self.load_batch(list(range(len(self.image_paths))), use_cache)
    
    def iterate_batches(
        self,
        batch_size: int,
        shuffle: bool = False
    ) -> Generator[Tuple[List[np.ndarray], List[int]], None, None]:
        """
        Iterate through dataset in batches.
        
        Args:
            batch_size: Number of images per batch
            shuffle: Whether to shuffle indices
            
        Yields:
            Tuple of (batch_images, batch_indices)
        """
        indices = list(range(len(self.image_paths)))
        
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images = self.load_batch(batch_indices, use_cache=False)
            yield batch_images, batch_indices
    
    def get_image_path(self, idx: int) -> Path:
        """Get path to image by index"""
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range")
        return self.image_paths[idx]
    
    def get_image_name(self, idx: int) -> str:
        """Get image filename by index"""
        return self.image_paths[idx].name
    
    def split_dataset(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True,
        random_seed: Optional[int] = 42
    ) -> Dict[str, List[int]]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            shuffle: Whether to shuffle before splitting
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing indices
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")
        
        n_images = len(self.image_paths)
        indices = np.arange(n_images)
        
        if shuffle:
            if random_seed is not None:
                np.random.seed(random_seed)
            np.random.shuffle(indices)
        
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        split = {
            'train': indices[:n_train].tolist(),
            'val': indices[n_train:n_train + n_val].tolist(),
            'test': indices[n_train + n_val:].tolist()
        }
        
        return split
    
    def load_metadata(self, metadata_path: Optional[str] = None):
        """
        Load metadata from JSON file.
        
        Args:
            metadata_path: Path to metadata JSON file
        """
        if metadata_path is None:
            metadata_path = self.data_dir / 'metadata.json'
        else:
            metadata_path = Path(metadata_path)
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata from {metadata_path}")
        else:
            print(f"Warning: Metadata file not found at {metadata_path}")
    
    def save_metadata(self, metadata_path: Optional[str] = None):
        """
        Save metadata to JSON file.
        
        Args:
            metadata_path: Path to save metadata
        """
        if metadata_path is None:
            metadata_path = self.data_dir / 'metadata.json'
        else:
            metadata_path = Path(metadata_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
    
    def get_dataset_statistics(self) -> Dict:
        """
        Compute statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'total_images': len(self.image_paths),
            'image_paths': [str(p) for p in self.image_paths[:10]],  # First 10
        }
        
        # Sample some images to get size information
        sample_size = min(10, len(self.image_paths))
        if sample_size > 0:
            sample_indices = np.linspace(0, len(self.image_paths) - 1, sample_size, dtype=int)
            shapes = []
            
            for idx in sample_indices:
                image = self.load_image(idx, use_cache=False)
                if image is not None:
                    shapes.append(image.shape)
            
            if shapes:
                heights = [s[0] for s in shapes]
                widths = [s[1] for s in shapes]
                
                stats['image_shapes'] = {
                    'height': {
                        'mean': float(np.mean(heights)),
                        'std': float(np.std(heights)),
                        'min': int(np.min(heights)),
                        'max': int(np.max(heights))
                    },
                    'width': {
                        'mean': float(np.mean(widths)),
                        'std': float(np.std(widths)),
                        'min': int(np.min(widths)),
                        'max': int(np.max(widths))
                    }
                }
        
        return stats
    
    def _update_cache(self, key: str, value: np.ndarray):
        """Update cache with LRU policy"""
        if len(self._cache) >= self.cache_size:
            # Remove oldest item
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = value
    
    def clear_cache(self):
        """Clear image cache"""
        self._cache = {}
    
    def __len__(self) -> int:
        """Return number of images in dataset"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> np.ndarray:
        """Get image by index using bracket notation"""
        return self.load_image(idx)
