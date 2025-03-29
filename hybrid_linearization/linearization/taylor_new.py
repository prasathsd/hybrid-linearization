import numpy as np
from typing import Tuple, List, Dict, Union
from scipy.misc import derivative
from sklearn.preprocessing import StandardScaler

class TaylorLinearization:
    def __init__(self,
                 max_order: int = 4,
                 epsilon: float = 1e-6,
                 adaptive_threshold: float = 0.01):
        """
        Initialize the Taylor series linearization framework.
        
        Args:
            max_order: Maximum order of Taylor series expansion
            epsilon: Small constant for numerical stability
            adaptive_threshold: Threshold for adaptive order selection
        """
        self.max_order = max_order
        self.epsilon = epsilon
        self.adaptive_threshold = adaptive_threshold
        self.scaler = StandardScaler()
        
    def compute_derivatives(self, 
                          func: callable,
                          x: np.ndarray,
                          order: int) -> List[np.ndarray]:
        """
        Compute derivatives up to specified order at point x.
        
        Args:
            func: Function to compute derivatives of
            x: Point to compute derivatives at
            order: Maximum order of derivatives
            
        Returns:
            List of derivatives from 0th to nth order
        """
        derivatives = []
        for n in range(order + 1):
            if n == 0:
                derivatives.append(func(x))
            else:
                # Compute nth derivative using finite differences
                d = derivative(func, x, n=n, dx=self.epsilon)
                derivatives.append(d)
        return derivatives 