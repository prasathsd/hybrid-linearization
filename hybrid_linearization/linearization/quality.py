import numpy as np
from typing import Dict, Tuple, List
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist

class LinearizationQualityMetrics:
    def __init__(self):
        """Initialize the linearization quality metrics calculator."""
        pass
    
    def compute_separability_metrics(self,
                                  features: np.ndarray,
                                  labels: np.ndarray) -> Dict[str, float]:
        """
        Compute various separability metrics for the feature space.
        
        Args:
            features: Feature matrix
            labels: Class labels
            
        Returns:
            Dictionary of separability metrics
        """
        # Handle NaN values by replacing them with mean values
        features = np.where(np.isnan(features),
                          np.nanmean(features, axis=0),
                          features)
        
        # Compute silhouette score
        try:
            silhouette = silhouette_score(features, labels)
        except:
            silhouette = 0.0
        
        # Compute Calinski-Harabasz score
        try:
            calinski_harabasz = calinski_harabasz_score(features, labels)
        except:
            calinski_harabasz = 0.0
        
        # Compute Davies-Bouldin score
        try:
            davies_bouldin = davies_bouldin_score(features, labels)
        except:
            davies_bouldin = 0.0
        
        # Compute class separation
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)
        class_separations = []
        
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                class_i_mask = labels == unique_classes[i]
                class_j_mask = labels == unique_classes[j]
                
                class_i_features = features[class_i_mask]
                class_j_features = features[class_j_mask]
                
                # Compute mean distance between classes
                mean_dist = np.mean(np.abs(
                    np.mean(class_i_features, axis=0) -
                    np.mean(class_j_features, axis=0)
                ))
                
                class_separations.append(mean_dist)
        
        mean_separation = np.mean(class_separations) if class_separations else 0.0
        
        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'mean_class_separation': mean_separation
        }
    
    def compute_linearity_metrics(self,
                               features: np.ndarray,
                               labels: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics that measure the linearity of the feature space.
        
        Args:
            features: Feature matrix
            labels: Class labels
            
        Returns:
            Dictionary of linearity metrics
        """
        # Handle NaN values by replacing them with mean values
        features = np.where(np.isnan(features),
                          np.nanmean(features, axis=0),
                          features)
        
        # Compute class centroids
        unique_labels = np.unique(labels)
        centroids = []
        for label in unique_labels:
            mask = labels == label
            centroid = np.mean(features[mask], axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        
        # Compute pairwise distances between centroids
        centroid_distances = cdist(centroids, centroids)
        
        # Compute linearity score based on centroid alignment
        linearity_score = self._compute_centroid_alignment(centroids)
        
        # Compute feature correlation matrix
        correlation_matrix = np.corrcoef(features.T)
        
        # Compute feature independence score
        independence_score = self._compute_feature_independence(correlation_matrix)
        
        return {
            'linearity_score': linearity_score,
            'independence_score': independence_score,
            'centroid_distances': centroid_distances
        }
    
    def compute_stability_metrics(self,
                               features: np.ndarray,
                               labels: np.ndarray,
                               n_bootstrap: int = 100) -> Dict[str, float]:
        """
        Compute stability metrics for the feature space.
        
        Args:
            features: Feature matrix
            labels: Class labels
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            Dictionary of stability metrics
        """
        # Bootstrap stability
        bootstrap_stability = self._compute_bootstrap_stability(
            features, labels, n_bootstrap
        )
        
        # Feature importance stability
        importance_stability = self._compute_feature_importance_stability(
            features, labels, n_bootstrap
        )
        
        # Class boundary stability
        boundary_stability = self._compute_boundary_stability(
            features, labels, n_bootstrap
        )
        
        return {
            'bootstrap_stability': bootstrap_stability,
            'importance_stability': importance_stability,
            'boundary_stability': boundary_stability
        }
    
    def _compute_davies_bouldin_score(self,
                                    features: np.ndarray,
                                    labels: np.ndarray) -> float:
        """Compute Davies-Bouldin score for cluster separation."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Compute cluster centers and average distances
        centers = []
        avg_distances = []
        for label in unique_labels:
            mask = labels == label
            cluster_points = features[mask]
            center = np.mean(cluster_points, axis=0)
            centers.append(center)
            avg_distances.append(np.mean(cdist(cluster_points, [center])))
        
        centers = np.array(centers)
        
        # Compute Davies-Bouldin score
        score = 0
        for i in range(n_clusters):
            max_ratio = 0
            for j in range(n_clusters):
                if i != j:
                    ratio = (avg_distances[i] + avg_distances[j]) / \
                           cdist([centers[i]], [centers[j]])[0][0]
                    max_ratio = max(max_ratio, ratio)
            score += max_ratio
        
        return score / n_clusters
    
    def _compute_fisher_ratio(self,
                            features: np.ndarray,
                            labels: np.ndarray) -> float:
        """Compute Fisher discriminant ratio."""
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # Compute overall mean
        mean_total = np.mean(features, axis=0)
        
        # Compute between-class scatter
        between_scatter = np.zeros((features.shape[1], features.shape[1]))
        for label in unique_labels:
            mask = labels == label
            class_mean = np.mean(features[mask], axis=0)
            diff = class_mean - mean_total
            between_scatter += np.outer(diff, diff) * np.sum(mask)
        
        # Compute within-class scatter
        within_scatter = np.zeros((features.shape[1], features.shape[1]))
        for label in unique_labels:
            mask = labels == label
            class_points = features[mask]
            class_mean = np.mean(class_points, axis=0)
            diff = class_points - class_mean
            within_scatter += diff.T @ diff
        
        # Compute Fisher ratio
        if np.linalg.det(within_scatter) != 0:
            fisher_ratio = np.trace(between_scatter @ np.linalg.inv(within_scatter))
        else:
            fisher_ratio = 0
            
        return fisher_ratio
    
    def _compute_centroid_alignment(self, centroids: np.ndarray) -> float:
        """Compute how well centroids align in a linear fashion."""
        # Compute principal components
        pca = np.linalg.svd(centroids - np.mean(centroids, axis=0))
        
        # Compute ratio of first eigenvalue to sum of all eigenvalues
        eigenvalues = pca[1]
        alignment_score = eigenvalues[0] / np.sum(eigenvalues)
        
        return alignment_score
    
    def _compute_feature_independence(self,
                                    correlation_matrix: np.ndarray) -> float:
        """Compute feature independence score based on correlation matrix."""
        # Remove diagonal elements
        mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
        off_diagonal_correlations = correlation_matrix[mask]
        
        # Compute independence score (lower correlation = higher independence)
        independence_score = 1 - np.mean(np.abs(off_diagonal_correlations))
        
        return independence_score
    
    def _compute_bootstrap_stability(self,
                                   features: np.ndarray,
                                   labels: np.ndarray,
                                   n_bootstrap: int) -> float:
        """Compute stability of feature space under bootstrap sampling."""
        stability_scores = []
        
        for _ in range(n_bootstrap):
            # Generate bootstrap indices
            indices = np.random.choice(
                len(features), size=len(features), replace=True
            )
            
            # Compute feature space for bootstrap sample
            bootstrap_features = features[indices]
            bootstrap_labels = labels[indices]
            
            # Compute separability metrics for bootstrap sample
            metrics = self.compute_separability_metrics(
                bootstrap_features, bootstrap_labels
            )
            
            stability_scores.append(metrics['silhouette_score'])
        
        return np.mean(stability_scores)
    
    def _compute_feature_importance_stability(self,
                                            features: np.ndarray,
                                            labels: np.ndarray,
                                            n_bootstrap: int) -> float:
        """Compute stability of feature importance rankings."""
        n_features = features.shape[1]
        importance_rankings = []
        
        for _ in range(n_bootstrap):
            # Generate bootstrap indices
            indices = np.random.choice(
                len(features), size=len(features), replace=True
            )
            
            # Compute feature importance for bootstrap sample
            bootstrap_features = features[indices]
            bootstrap_labels = labels[indices]
            
            # Compute feature importance using Fisher ratio
            importance = []
            for i in range(n_features):
                fisher_ratio = self._compute_fisher_ratio(
                    bootstrap_features[:, [i]], bootstrap_labels
                )
                importance.append(fisher_ratio)
            
            # Store ranking
            ranking = np.argsort(importance)[::-1]
            importance_rankings.append(ranking)
        
        # Compute stability as average Spearman correlation between rankings
        stability = 0
        for i in range(n_bootstrap):
            for j in range(i + 1, n_bootstrap):
                correlation = self._spearman_correlation(
                    importance_rankings[i], importance_rankings[j]
                )
                stability += correlation
        
        return stability / (n_bootstrap * (n_bootstrap - 1) / 2)
    
    def _compute_boundary_stability(self,
                                  features: np.ndarray,
                                  labels: np.ndarray,
                                  n_bootstrap: int) -> float:
        """Compute stability of class boundaries."""
        boundary_stabilities = []
        
        for _ in range(n_bootstrap):
            # Generate bootstrap indices
            indices = np.random.choice(
                len(features), size=len(features), replace=True
            )
            
            # Compute feature space for bootstrap sample
            bootstrap_features = features[indices]
            bootstrap_labels = labels[indices]
            
            # Compute class boundaries using SVM
            from sklearn.svm import SVC
            svm = SVC(kernel='linear')
            svm.fit(bootstrap_features, bootstrap_labels)
            
            # Compute boundary stability
            boundary_stability = np.mean(np.abs(svm.coef_))
            boundary_stabilities.append(boundary_stability)
        
        return np.mean(boundary_stabilities)
    
    def _spearman_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Spearman correlation between two rankings."""
        n = len(x)
        x_ranks = np.argsort(x) + 1
        y_ranks = np.argsort(y) + 1
        
        d = x_ranks - y_ranks
        correlation = 1 - (6 * np.sum(d**2)) / (n * (n**2 - 1))
        
        return correlation 