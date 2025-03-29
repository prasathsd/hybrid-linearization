import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor

class CrossModalEnhancer:
    def __init__(self,
                 n_components: int = 100,
                 learning_rate: float = 0.001,
                 max_iter: int = 1000):
        """
        Initialize the cross-modal feature enhancement system.
        
        Args:
            n_components: Number of components for dimensionality reduction
            learning_rate: Learning rate for neural network
            max_iter: Maximum number of iterations for training
        """
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
        # Initialize components
        self.image_scaler = StandardScaler()
        self.speech_scaler = StandardScaler()
        self.image_pca = PCA(n_components=n_components)
        self.speech_pca = PCA(n_components=n_components)
        self.image_to_speech = None
        self.speech_to_image = None
        
    def prepare_features(self,
                        image_features: np.ndarray,
                        speech_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for cross-modal mapping.
        
        Args:
            image_features: Image feature matrix
            speech_features: Speech feature matrix
            
        Returns:
            Tuple of (prepared image features, prepared speech features)
        """
        # Scale features
        image_scaled = self.image_scaler.fit_transform(image_features)
        speech_scaled = self.speech_scaler.fit_transform(speech_features)
        
        # Apply PCA
        image_reduced = self.image_pca.fit_transform(image_scaled)
        speech_reduced = self.speech_pca.fit_transform(speech_scaled)
        
        return image_reduced, speech_reduced
    
    def train_mapping(self,
                     image_features: np.ndarray,
                     speech_features: np.ndarray) -> None:
        """
        Train the cross-modal mapping networks.
        
        Args:
            image_features: Image feature matrix
            speech_features: Speech feature matrix
        """
        # Prepare features
        image_reduced, speech_reduced = self.prepare_features(
            image_features, speech_features
        )
        
        # Train image to speech mapping
        self.image_to_speech = MLPRegressor(
            hidden_layer_sizes=(256, 128),
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=42
        )
        self.image_to_speech.fit(image_reduced, speech_reduced)
        
        # Train speech to image mapping
        self.speech_to_image = MLPRegressor(
            hidden_layer_sizes=(256, 128),
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=42
        )
        self.speech_to_image.fit(speech_reduced, image_reduced)
    
    def map_image_to_speech(self,
                          image_features: np.ndarray) -> np.ndarray:
        """
        Map image features to speech feature space.
        
        Args:
            image_features: Image feature matrix
            
        Returns:
            Mapped speech features
        """
        if self.image_to_speech is None:
            raise ValueError("Mapping has not been trained yet.")
            
        # Scale and reduce image features
        image_scaled = self.image_scaler.transform(image_features)
        image_reduced = self.image_pca.transform(image_scaled)
        
        # Map to speech space
        speech_reduced = self.image_to_speech.predict(image_reduced)
        
        # Transform back to original space
        speech_features = self.speech_pca.inverse_transform(speech_reduced)
        speech_features = self.speech_scaler.inverse_transform(speech_features)
        
        return speech_features
    
    def map_speech_to_image(self,
                          speech_features: np.ndarray) -> np.ndarray:
        """
        Map speech features to image feature space.
        
        Args:
            speech_features: Speech feature matrix
            
        Returns:
            Mapped image features
        """
        if self.speech_to_image is None:
            raise ValueError("Mapping has not been trained yet.")
            
        # Scale and reduce speech features
        speech_scaled = self.speech_scaler.transform(speech_features)
        speech_reduced = self.speech_pca.transform(speech_scaled)
        
        # Map to image space
        image_reduced = self.speech_to_image.predict(speech_reduced)
        
        # Transform back to original space
        image_features = self.image_pca.inverse_transform(image_reduced)
        image_features = self.image_scaler.inverse_transform(image_features)
        
        return image_features
    
    def evaluate_mapping(self,
                        image_features: np.ndarray,
                        speech_features: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the quality of cross-modal mapping.
        
        Args:
            image_features: Original image features
            speech_features: Original speech features
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Map features
        mapped_speech = self.map_image_to_speech(image_features)
        mapped_image = self.map_speech_to_image(speech_features)
        
        # Compute reconstruction errors
        speech_error = np.mean(np.abs(speech_features - mapped_speech))
        image_error = np.mean(np.abs(image_features - mapped_image))
        
        # Compute feature correlation
        speech_correlation = np.mean(np.corrcoef(
            speech_features.flatten(),
            mapped_speech.flatten()
        ))
        image_correlation = np.mean(np.corrcoef(
            image_features.flatten(),
            mapped_image.flatten()
        ))
        
        return {
            'speech_reconstruction_error': speech_error,
            'image_reconstruction_error': image_error,
            'speech_feature_correlation': speech_correlation,
            'image_feature_correlation': image_correlation
        }
    
    def enhance_features(self,
                        image_features: np.ndarray,
                        speech_features: np.ndarray,
                        alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhance features using cross-modal information.
        
        Args:
            image_features: Original image features
            speech_features: Original speech features
            alpha: Mixing parameter for feature enhancement
            
        Returns:
            Tuple of (enhanced image features, enhanced speech features)
        """
        # Map features
        mapped_speech = self.map_image_to_speech(image_features)
        mapped_image = self.map_speech_to_image(speech_features)
        
        # Enhance features
        enhanced_speech = alpha * speech_features + (1 - alpha) * mapped_speech
        enhanced_image = alpha * image_features + (1 - alpha) * mapped_image
        
        return enhanced_image, enhanced_speech
    
    def visualize_mapping(self,
                         image_features: np.ndarray,
                         speech_features: np.ndarray,
                         n_samples: int = 1000) -> None:
        """
        Visualize the cross-modal mapping using t-SNE.
        
        Args:
            image_features: Image feature matrix
            speech_features: Speech feature matrix
            n_samples: Number of samples to visualize
        """
        import matplotlib.pyplot as plt
        
        # Select random samples
        indices = np.random.choice(
            len(image_features), size=min(n_samples, len(image_features)), replace=False
        )
        image_samples = image_features[indices]
        speech_samples = speech_features[indices]
        
        # Map features
        mapped_speech = self.map_image_to_speech(image_samples)
        mapped_image = self.map_speech_to_image(speech_samples)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        
        # Transform features
        image_tsne = tsne.fit_transform(image_samples)
        speech_tsne = tsne.fit_transform(speech_samples)
        mapped_speech_tsne = tsne.fit_transform(mapped_speech)
        mapped_image_tsne = tsne.fit_transform(mapped_image)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
        
        # Plot original features
        ax1.scatter(image_tsne[:, 0], image_tsne[:, 1], alpha=0.6)
        ax1.set_title('Original Image Features')
        
        ax2.scatter(speech_tsne[:, 0], speech_tsne[:, 1], alpha=0.6)
        ax2.set_title('Original Speech Features')
        
        # Plot mapped features
        ax3.scatter(mapped_speech_tsne[:, 0], mapped_speech_tsne[:, 1], alpha=0.6)
        ax3.set_title('Mapped Speech Features')
        
        ax4.scatter(mapped_image_tsne[:, 0], mapped_image_tsne[:, 1], alpha=0.6)
        ax4.set_title('Mapped Image Features')
        
        plt.tight_layout()
        plt.show()
    
    def get_mapping_info(self) -> Dict:
        """
        Get information about the trained mapping.
        
        Returns:
            Dictionary containing mapping information
        """
        return {
            'n_components': self.n_components,
            'image_pca_explained_variance': np.sum(self.image_pca.explained_variance_ratio_),
            'speech_pca_explained_variance': np.sum(self.speech_pca.explained_variance_ratio_),
            'mapping_trained': self.image_to_speech is not None
        } 