"""
Hybrid Linearization Pathways for Multi-Modal Classification

This package implements a novel approach to feature extraction and linearization
for multi-modal data classification, specifically focusing on image and speech data.
"""

from .features.image_features import ImageFeatureExtractor
from .features.speech_features import SpeechFeatureExtractor
from .linearization.taylor import TaylorLinearization
from .linearization.quality import LinearizationQualityMetrics
from .linearization.visualization import LinearizationVisualizer
from .models.svm_optimizer import SVMOptimizer
from .models.cross_modal import CrossModalEnhancer
from .evaluation.benchmarking import ModelBenchmark

__version__ = '0.1.0' 