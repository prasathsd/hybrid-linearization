"""
Feature extraction modules for multi-modal data.
"""

from .image_features import ImageFeatureExtractor
from .speech_features import SpeechFeatureExtractor

__all__ = ['ImageFeatureExtractor', 'SpeechFeatureExtractor'] 