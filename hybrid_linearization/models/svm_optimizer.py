import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score

class SVMOptimizer:
    def __init__(self,
                 kernel_types: List[str] = ['linear', 'rbf', 'poly'],
                 param_grid: Optional[Dict] = None):
        """
        Initialize the SVM optimizer with configurable kernel types and parameter grid.
        
        Args:
            kernel_types: List of kernel types to consider
            param_grid: Optional parameter grid for hyperparameter optimization
        """
        self.kernel_types = kernel_types
        self.param_grid = param_grid or {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'degree': [2, 3, 4],
            'class_weight': ['balanced', None]
        }
        
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_kernel = None
        self.best_params = None
        self.feature_importance = None
        
    def select_kernel(self,
                     features: np.ndarray,
                     labels: np.ndarray,
                     cv: int = 5) -> str:
        """
        Select the optimal kernel type based on data characteristics.
        
        Args:
            features: Feature matrix
            labels: Class labels
            cv: Number of cross-validation folds
            
        Returns:
            Selected kernel type
        """
        # Compute data characteristics
        n_samples, n_features = features.shape
        n_classes = len(np.unique(labels))
        
        # Initialize kernel scores
        kernel_scores = {}
        
        for kernel in self.kernel_types:
            if kernel == 'linear':
                # Linear kernel is preferred for high-dimensional data
                if n_features > n_samples:
                    kernel_scores[kernel] = 1.0
                else:
                    # Evaluate linear kernel performance
                    model = LinearSVC(max_iter=1000)
                    scores = self._cross_validate(model, features, labels, cv)
                    kernel_scores[kernel] = np.mean(scores)
                    
            elif kernel == 'rbf':
                # RBF kernel is preferred for non-linear data
                if n_features < n_samples:
                    # Evaluate RBF kernel performance
                    model = SVC(kernel='rbf')
                    scores = self._cross_validate(model, features, labels, cv)
                    kernel_scores[kernel] = np.mean(scores)
                else:
                    kernel_scores[kernel] = 0.0
                    
            elif kernel == 'poly':
                # Polynomial kernel is preferred for moderate non-linearity
                if n_features < n_samples and n_classes > 2:
                    # Evaluate polynomial kernel performance
                    model = SVC(kernel='poly')
                    scores = self._cross_validate(model, features, labels, cv)
                    kernel_scores[kernel] = np.mean(scores)
                else:
                    kernel_scores[kernel] = 0.0
        
        # Select kernel with highest score
        best_kernel = max(kernel_scores.items(), key=lambda x: x[1])[0]
        return best_kernel
    
    def optimize_hyperparameters(self,
                               features: np.ndarray,
                               labels: np.ndarray,
                               kernel: str,
                               cv: int = 5) -> Dict:
        """
        Optimize hyperparameters for the selected kernel.
        
        Args:
            features: Feature matrix
            labels: Class labels
            kernel: Selected kernel type
            cv: Number of cross-validation folds
            
        Returns:
            Optimized hyperparameters
        """
        # Prepare parameter grid for the selected kernel
        if kernel == 'linear':
            param_grid = {k: v for k, v in self.param_grid.items() 
                         if k in ['C', 'class_weight']}
            model = LinearSVC(max_iter=1000)
        else:
            param_grid = self.param_grid
            model = SVC(kernel=kernel)
        
        # Create scoring dictionary
        scoring = {
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1': make_scorer(f1_score, average='weighted'),
            'accuracy': make_scorer(accuracy_score)
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            refit='f1',
            n_jobs=-1
        )
        
        grid_search.fit(features, labels)
        
        return grid_search.best_params_
    
    def train(self,
             features: np.ndarray,
             labels: np.ndarray,
             cv: int = 5) -> None:
        """
        Train SVM model with optimal parameters.
        
        Args:
            features: Feature matrix
            labels: Target labels
            cv: Number of cross-validation folds
        """
        # Handle NaN values by replacing them with mean values
        features = np.where(np.isnan(features),
                          np.nanmean(features, axis=0),
                          features)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Select best kernel
        self.best_kernel = self.select_kernel(features_scaled, labels, cv)
        
        # Train final model
        self.model = SVC(kernel=self.best_kernel)
        self.model.fit(features_scaled, labels)
        
        # Store training results
        self.training_results = {
            'features': features_scaled,
            'labels': labels,
            'kernel': self.best_kernel
        }
        
        # Compute feature importance if using linear kernel
        if self.best_kernel == 'linear':
            self.feature_importance = np.abs(self.model.coef_[0])
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            features: Feature matrix
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Handle NaN values by replacing them with mean values
        features = np.where(np.isnan(features),
                          np.nanmean(features, axis=0),
                          features)
        
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
    
    def evaluate(self,
                features: np.ndarray,
                labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            features: Feature matrix
            labels: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(features)
        
        metrics = {
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
            'f1': f1_score(labels, predictions, average='weighted'),
            'accuracy': accuracy_score(labels, predictions)
        }
        
        return metrics
    
    def _cross_validate(self,
                       model: Union[LinearSVC, SVC],
                       features: np.ndarray,
                       labels: np.ndarray,
                       cv: int) -> np.ndarray:
        """
        Perform cross-validation and return scores.
        
        Args:
            model: SVM model
            features: Feature matrix
            labels: Class labels
            cv: Number of cross-validation folds
            
        Returns:
            Array of cross-validation scores
        """
        from sklearn.model_selection import cross_val_score
        
        scoring = make_scorer(f1_score, average='weighted')
        scores = cross_val_score(model, features, labels, cv=cv, scoring=scoring)
        
        return scores
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores if using linear kernel.
        
        Returns:
            Array of feature importance scores or None if not using linear kernel
        """
        return self.feature_importance
    
    def get_model_info(self) -> Dict:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'kernel': self.best_kernel,
            'parameters': self.best_params,
            'feature_importance': self.feature_importance is not None
        } 