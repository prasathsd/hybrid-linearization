import numpy as np
import cv2
from scipy import ndimage
from typing import Tuple, List, Dict, Union
from skimage.feature import hog
from skimage.feature import local_binary_pattern

class ImageFeatureExtractor:
    def __init__(self):
        """Initialize the feature extractor with default parameters."""
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2)
        }
        self.lbp_params = {
            'P': 8,  # number of circularly symmetric neighbor set points
            'R': 1,  # radius of circle
            'method': 'uniform'
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for feature extraction.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Ensure image is float32 and in range [0, 1]
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        if image.max() > 1.0:
            image /= 255.0
            
        return image
    
    def extract_hog(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Histogram of Oriented Gradients (HOG) features.
        
        Args:
            image: Input image
            
        Returns:
            HOG features as 1D array
        """
        # Preprocess image
        image = self.preprocess_image(image)
        
        # Extract HOG features
        features = hog(
            image,
            orientations=self.hog_params['orientations'],
            pixels_per_cell=self.hog_params['pixels_per_cell'],
            cells_per_block=self.hog_params['cells_per_block'],
            feature_vector=True
        )
        
        return features
    
    def extract_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Local Binary Pattern (LBP) features.
        
        Args:
            image: Input image
            
        Returns:
            LBP features as 1D array
        """
        # Preprocess image
        image = self.preprocess_image(image)
        
        # Extract LBP features
        lbp = local_binary_pattern(
            image,
            P=self.lbp_params['P'],
            R=self.lbp_params['R'],
            method=self.lbp_params['method']
        )
        
        # Compute histogram of LBP values
        n_bins = self.lbp_params['P'] + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-7  # Normalize
        
        return hist
    
    def extract_geometric_invariants(self, image: np.ndarray) -> np.ndarray:
        """
        Extract geometric invariant features (moments, etc.).
        
        Args:
            image: Input image
            
        Returns:
            Geometric features as 1D array
        """
        # Preprocess image
        image = self.preprocess_image(image)
        
        # Calculate moments
        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Log transform Hu moments (helps with numerical stability)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-7)
        
        return hu_moments
    
    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract all features from the image.
        
        Args:
            image: Input image
            
        Returns:
            Combined feature vector
        """
        # Extract individual feature sets
        hog_features = self.extract_hog(image)
        lbp_features = self.extract_lbp(image)
        geometric_features = self.extract_geometric_invariants(image)
        
        # Combine all features
        combined_features = np.concatenate([
            hog_features,
            lbp_features,
            geometric_features
        ])
        
        return combined_features 