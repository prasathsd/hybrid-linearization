import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import os
import logging

from .features.image_features import ImageFeatureExtractor
from .features.speech_features import SpeechFeatureExtractor
from .linearization.taylor import TaylorLinearization
from .linearization.quality import LinearizationQualityMetrics
from .linearization.visualization import LinearizationVisualizer
from .models.svm_optimizer import SVMOptimizer
from .models.cross_modal import CrossModalEnhancer
from .evaluation.benchmarking import ModelBenchmark

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_mnist_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset.
    
    Args:
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Use only 1000 samples for faster processing
    train_dataset.data = train_dataset.data[:1000]
    train_dataset.targets = train_dataset.targets[:1000]
    test_dataset.data = test_dataset.data[:200]
    test_dataset.targets = test_dataset.targets[:200]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def extract_features(train_loader: DataLoader,
                    test_loader: DataLoader) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Extract features from the datasets.
    
    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        
    Returns:
        Tuple of (train_features, test_features)
    """
    # Initialize feature extractors
    image_extractor = ImageFeatureExtractor()
    
    # Extract features
    train_features = {'image': [], 'labels': []}
    test_features = {'image': [], 'labels': []}
    
    # Process training data
    logger.info("Extracting features from training data...")
    for images, labels in train_loader:
        # Convert to numpy arrays
        images = images.numpy()
        labels = labels.numpy()
        
        # Extract features for each image
        for image in images:
            # MNIST images are (1, 28, 28), convert to (28, 28)
            image = np.squeeze(image)
            features = image_extractor.extract_all_features(image)
            train_features['image'].append(features)
        train_features['labels'].extend(labels)
    
    # Process test data
    logger.info("Extracting features from test data...")
    for images, labels in test_loader:
        # Convert to numpy arrays
        images = images.numpy()
        labels = labels.numpy()
        
        # Extract features for each image
        for image in images:
            # MNIST images are (1, 28, 28), convert to (28, 28)
            image = np.squeeze(image)
            features = image_extractor.extract_all_features(image)
            test_features['image'].append(features)
        test_features['labels'].extend(labels)
    
    # Convert to numpy arrays
    train_features['image'] = np.array(train_features['image'])
    train_features['labels'] = np.array(train_features['labels'])
    test_features['image'] = np.array(test_features['image'])
    test_features['labels'] = np.array(test_features['labels'])
    
    return train_features, test_features

def main():
    """Main function to demonstrate the hybrid linearization framework."""
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Load data
    logger.info("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data()
    
    # Extract features
    train_features, test_features = extract_features(train_loader, test_loader)
    
    # Initialize components
    logger.info("Initializing framework components...")
    taylor = TaylorLinearization()
    quality_metrics = LinearizationQualityMetrics()
    visualizer = LinearizationVisualizer()
    svm_optimizer = SVMOptimizer()
    cross_modal = CrossModalEnhancer()
    benchmark = ModelBenchmark()
    
    # Linearize feature space
    logger.info("Linearizing feature space...")
    linearization_results = taylor.linearize_feature_space(
        train_features['image'],
        train_features['labels']
    )
    
    # Evaluate linearization quality
    logger.info("Evaluating linearization quality...")
    quality_results = quality_metrics.compute_separability_metrics(
        linearization_results['linearized_features'],
        train_features['labels']
    )
    
    # Visualize results
    logger.info("Generating visualizations...")
    visualizer.plot_feature_space(
        train_features['image'],
        train_features['labels'],
        "Original Feature Space"
    )
    visualizer.plot_linearization_progress(
        train_features['image'],
        linearization_results['linearized_features'],
        train_features['labels'],
        "Linearization Progress"
    )
    visualizer.plot_quality_metrics(quality_results)
    
    # Train SVM model
    logger.info("Training SVM model...")
    svm_optimizer.train(
        linearization_results['linearized_features'],
        train_features['labels']
    )
    
    # Evaluate model
    logger.info("Evaluating model performance...")
    model_metrics = svm_optimizer.evaluate(
        linearization_results['linearized_features'],
        train_features['labels']
    )
    
    # Benchmark models
    logger.info("Running model benchmarks...")
    benchmark.benchmark_linearization(
        svm_optimizer,
        linearization_results['linearized_features'],
        train_features['labels'],
        test_features['image'],
        test_features['labels']
    )
    
    # Generate benchmark report
    logger.info("Generating benchmark report...")
    report = benchmark.generate_report()
    
    # Save results
    logger.info("Saving results...")
    with open('output/benchmark_report.txt', 'w') as f:
        f.write(report)
    
    logger.info("Framework demonstration completed successfully!")

if __name__ == "__main__":
    main() 