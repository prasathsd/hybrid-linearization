import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, List, Optional

class LinearizationVisualizer:
    """Class for visualizing linearization results."""
    
    def __init__(self):
        """Initialize the visualizer with default style settings."""
        plt.style.use('default')  # Use default matplotlib style
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def plot_feature_space(self, features: np.ndarray, labels: np.ndarray,
                          title: str = "Feature Space Visualization") -> None:
        """
        Plot feature space using t-SNE for dimensionality reduction.
        
        Args:
            features: Feature matrix
            labels: Labels array
            title: Plot title
        """
        # Handle NaN values by replacing them with mean values
        features = np.where(np.isnan(features),
                          np.nanmean(features, axis=0),
                          features)
        
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        for i in range(10):  # Assuming 10 classes (0-9)
            mask = labels == i
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[self.colors[i]], label=f'Class {i}',
                       alpha=0.6)
        
        plt.title(title)
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"output/{title.lower().replace(' ', '_')}.png")
        plt.close()
    
    def plot_linearization_progress(self, original_features: np.ndarray,
                                  linearized_features: np.ndarray,
                                  labels: np.ndarray,
                                  title: str = "Linearization Progress") -> None:
        """
        Plot original vs linearized feature spaces side by side.
        
        Args:
            original_features: Original feature matrix
            linearized_features: Linearized feature matrix
            labels: Labels array
            title: Plot title
        """
        # Handle NaN values by replacing them with mean values
        original_features = np.where(np.isnan(original_features),
                                   np.nanmean(original_features, axis=0),
                                   original_features)
        linearized_features = np.where(np.isnan(linearized_features),
                                     np.nanmean(linearized_features, axis=0),
                                     linearized_features)
        
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        original_2d = tsne.fit_transform(original_features)
        linearized_2d = tsne.fit_transform(linearized_features)
        
        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot original features
        for i in range(10):
            mask = labels == i
            ax1.scatter(original_2d[mask, 0], original_2d[mask, 1],
                       c=[self.colors[i]], label=f'Class {i}',
                       alpha=0.6)
        ax1.set_title("Original Feature Space")
        ax1.set_xlabel("t-SNE Component 1")
        ax1.set_ylabel("t-SNE Component 2")
        ax1.legend()
        
        # Plot linearized features
        for i in range(10):
            mask = labels == i
            ax2.scatter(linearized_2d[mask, 0], linearized_2d[mask, 1],
                       c=[self.colors[i]], label=f'Class {i}',
                       alpha=0.6)
        ax2.set_title("Linearized Feature Space")
        ax2.set_xlabel("t-SNE Component 1")
        ax2.set_ylabel("t-SNE Component 2")
        ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(f"output/{title.lower().replace(' ', '_')}.png")
        plt.close()
    
    def plot_quality_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Plot quality metrics as a bar chart.
        
        Args:
            metrics: Dictionary of quality metrics
        """
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        x = np.arange(len(metrics))
        plt.bar(x, list(metrics.values()))
        
        # Customize plot
        plt.title("Linearization Quality Metrics")
        plt.xticks(x, list(metrics.keys()), rotation=45)
        plt.ylabel("Score")
        
        # Add value labels on top of bars
        for i, v in enumerate(metrics.values()):
            plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig("output/quality_metrics.png")
        plt.close()