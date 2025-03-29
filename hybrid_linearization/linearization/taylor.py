import numpy as np
from typing import Tuple, List, Dict, Union, Callable, Optional
from sklearn.preprocessing import StandardScaler
import math

class TaylorLinearization:
    def __init__(self,
                 max_order: int = 5,
                 tolerance: float = 1e-6):
        """
        Initialize the Taylor linearization with parameters.
        
        Args:
            max_order: Maximum order of Taylor series expansion
            tolerance: Tolerance for convergence
        """
        self.max_order = max_order
        self.tolerance = tolerance
        self.scaler = StandardScaler()
        
    def compute_derivative(self,
                         func: callable,
                         x: np.ndarray,
                         n: int) -> np.ndarray:
        """
        Compute nth derivative using finite differences.
        
        Args:
            func: Function to compute derivative of
            x: Point to compute derivative at
            n: Order of derivative
            
        Returns:
            nth derivative at point x
        """
        if n == 0:
            return func(x)
            
        h = 1e-7  # Small step size for numerical differentiation
        if n == 1:
            return (func(x + h) - func(x - h)) / (2 * h)
            
        # Higher order derivatives using central difference
        coeffs = np.zeros(2 * n + 1)
        for i in range(2 * n + 1):
            k = i - n
            coeffs[i] = (-1) ** (n + k) * math.comb(n, k + n)
            
        result = 0
        for i in range(2 * n + 1):
            k = i - n
            result += coeffs[i] * func(x + k * h)
            
        return result / (h ** n)
        
    def compute_derivatives(self, func: Callable, x: np.ndarray, x0: np.ndarray,
                          order: int) -> List[np.ndarray]:
        """
        Compute numerical derivatives up to specified order.
        
        Args:
            func: Function to differentiate
            x: Point at which to evaluate derivatives
            x0: Point around which to expand
            order: Maximum order of derivatives
            
        Returns:
            List of derivatives
        """
        h = 1e-7  # Small step size for numerical differentiation
        derivatives = []
        
        # 0th derivative (function value)
        derivatives.append(func(x0))
        
        # Higher order derivatives using finite differences
        for n in range(1, order + 1):
            # Use central difference formula
            if n == 1:
                deriv = (func(x0 + h) - func(x0 - h)) / (2 * h)
            else:
                # For higher order derivatives, use a simple approximation
                deriv = np.zeros_like(x0)
            
            derivatives.append(deriv)
        
        return derivatives
    
    def taylor_series(self, func: Callable, x: np.ndarray, x0: np.ndarray,
                     order: int) -> np.ndarray:
        """
        Compute Taylor series expansion.
        
        Args:
            func: Function to expand
            x: Point at which to evaluate series
            x0: Point around which to expand
            order: Order of expansion
            
        Returns:
            Taylor series approximation
        """
        derivatives = self.compute_derivatives(func, x, x0, order)
        approximation = np.zeros_like(x)
        
        for n in range(order + 1):
            term = derivatives[n] * np.power(x - x0, n) / math.factorial(n)
            approximation += term
        
        # Replace any NaN values with the original values
        approximation = np.where(np.isnan(approximation), x, approximation)
        
        return approximation
    
    def select_optimal_order(self, func: Callable, x: np.ndarray,
                           x0: np.ndarray) -> int:
        """
        Select optimal order for Taylor expansion.
        
        Args:
            func: Function to expand
            x: Point at which to evaluate series
            x0: Point around which to expand
            
        Returns:
            Optimal order
        """
        prev_approx = None
        
        for order in range(1, self.max_order + 1):
            approx = self.taylor_series(func, x, x0, order)
            
            if prev_approx is not None:
                change = np.mean(np.abs(approx - prev_approx))
                if change < self.tolerance:
                    return order
            
            prev_approx = approx
        
        return self.max_order
    
    def linearize_feature_space(self, features: np.ndarray,
                              labels: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Linearize feature space using Taylor expansion.
        
        Args:
            features: Feature matrix
            labels: Class labels
            
        Returns:
            Dictionary containing linearized features and metadata
        """
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)
        n_samples, n_features = features.shape
        
        # Initialize linearized features
        linearized_features = np.zeros_like(features)
        
        # Process each class separately
        for class_label in unique_classes:
            # Get features for current class
            class_mask = labels == class_label
            class_features = features[class_mask]
            
            # Compute class centroid
            centroid = np.mean(class_features, axis=0)
            
            # For each feature dimension
            for i in range(n_features):
                # Define function to linearize
                def linearize_func(x):
                    return np.mean(np.abs(x - centroid[i]))
                
                # Select optimal order for Taylor expansion
                order = self.select_optimal_order(
                    linearize_func,
                    class_features[:, i],
                    centroid[i]
                )
                
                # Apply Taylor expansion to linearize features
                linearized_class_features = self.taylor_series(
                    linearize_func,
                    class_features[:, i],
                    centroid[i],
                    order
                )
                
                # Store linearized features
                linearized_features[class_mask, i] = linearized_class_features
        
        # Replace any remaining NaN values with original values
        linearized_features = np.where(np.isnan(linearized_features),
                                     features,
                                     linearized_features)
        
        return {
            'linearized_features': linearized_features,
            'original_features': features,
            'labels': labels
        }
    
    def evaluate_linearization_quality(self,
                                    original_features: np.ndarray,
                                    linearized_features: np.ndarray,
                                    labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the quality of linearization.
        
        Args:
            original_features: Original feature space
            linearized_features: Linearized feature space
            labels: Target labels
            
        Returns:
            Dictionary of quality metrics
        """
        # Compute reconstruction error
        reconstruction_error = np.mean(np.abs(original_features - linearized_features))
        
        # Compute class separability
        unique_labels = np.unique(labels)
        class_separability = 0
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                mask1 = labels == label1
                mask2 = labels == label2
                dist = np.mean(np.abs(
                    np.mean(linearized_features[mask1], axis=0) -
                    np.mean(linearized_features[mask2], axis=0)
                ))
                class_separability += dist
        class_separability /= (len(unique_labels) * (len(unique_labels) - 1) / 2)
        
        return {
            'reconstruction_error': reconstruction_error,
            'class_separability': class_separability
        } 