import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import time
import psutil
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import AutoModel, AutoTokenizer

class ModelBenchmark:
    def __init__(self,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the model benchmarking framework.
        
        Args:
            device: Device to run models on (cuda/cpu)
        """
        self.device = device
        self.results = {}
        
    def benchmark_linearization(self,
                              model: object,
                              features: np.ndarray,
                              labels: np.ndarray,
                              test_features: np.ndarray,
                              test_labels: np.ndarray) -> Dict:
        """
        Benchmark the linearization-based model.
        
        Args:
            model: Linearization model
            features: Training features
            labels: Training labels
            test_features: Test features
            test_labels: Test labels
            
        Returns:
            Dictionary of benchmark results
        """
        # Training time
        start_time = time.time()
        model.train(features, labels)
        training_time = time.time() - start_time
        
        # Memory usage during training
        training_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Inference time
        start_time = time.time()
        predictions = model.predict(test_features)
        inference_time = time.time() - start_time
        
        # Memory usage during inference
        inference_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Performance metrics
        metrics = {
            'precision': precision_score(test_labels, predictions, average='weighted'),
            'recall': recall_score(test_labels, predictions, average='weighted'),
            'f1': f1_score(test_labels, predictions, average='weighted'),
            'accuracy': accuracy_score(test_labels, predictions)
        }
        
        results = {
            'training_time': training_time,
            'inference_time': inference_time,
            'training_memory': training_memory,
            'inference_memory': inference_memory,
            **metrics
        }
        
        self.results['linearization'] = results
        return results
    
    def benchmark_cnn(self,
                     model: nn.Module,
                     train_loader: DataLoader,
                     test_loader: DataLoader,
                     n_epochs: int = 10) -> Dict:
        """
        Benchmark the CNN model.
        
        Args:
            model: CNN model
            train_loader: Training data loader
            test_loader: Test data loader
            n_epochs: Number of training epochs
            
        Returns:
            Dictionary of benchmark results
        """
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        
        # Training time and memory
        start_time = time.time()
        max_memory = 0
        
        for epoch in range(n_epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Track memory usage
                current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                max_memory = max(max_memory, current_memory)
        
        training_time = time.time() - start_time
        
        # Inference time and memory
        start_time = time.time()
        predictions = []
        true_labels = []
        
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(target.numpy())
        
        inference_time = time.time() - start_time
        inference_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Performance metrics
        metrics = {
            'precision': precision_score(true_labels, predictions, average='weighted'),
            'recall': recall_score(true_labels, predictions, average='weighted'),
            'f1': f1_score(true_labels, predictions, average='weighted'),
            'accuracy': accuracy_score(true_labels, predictions)
        }
        
        results = {
            'training_time': training_time,
            'inference_time': inference_time,
            'training_memory': max_memory,
            'inference_memory': inference_memory,
            **metrics
        }
        
        self.results['cnn'] = results
        return results
    
    def benchmark_transformer(self,
                            model_name: str,
                            train_loader: DataLoader,
                            test_loader: DataLoader,
                            n_epochs: int = 10) -> Dict:
        """
        Benchmark the Transformer model.
        
        Args:
            model_name: Name of the transformer model
            train_loader: Training data loader
            test_loader: Test data loader
            n_epochs: Number of training epochs
            
        Returns:
            Dictionary of benchmark results
        """
        # Load model and tokenizer
        model = AutoModel.from_pretrained(model_name).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters())
        
        # Training time and memory
        start_time = time.time()
        max_memory = 0
        
        for epoch in range(n_epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # Tokenize input
                inputs = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                target = target.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(**inputs)
                loss = criterion(outputs.logits, target)
                loss.backward()
                optimizer.step()
                
                # Track memory usage
                current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                max_memory = max(max_memory, current_memory)
        
        training_time = time.time() - start_time
        
        # Inference time and memory
        start_time = time.time()
        predictions = []
        true_labels = []
        
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                # Tokenize input
                inputs = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                pred = outputs.logits.argmax(dim=1)
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(target.numpy())
        
        inference_time = time.time() - start_time
        inference_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Performance metrics
        metrics = {
            'precision': precision_score(true_labels, predictions, average='weighted'),
            'recall': recall_score(true_labels, predictions, average='weighted'),
            'f1': f1_score(true_labels, predictions, average='weighted'),
            'accuracy': accuracy_score(true_labels, predictions)
        }
        
        results = {
            'training_time': training_time,
            'inference_time': inference_time,
            'training_memory': max_memory,
            'inference_memory': inference_memory,
            **metrics
        }
        
        self.results['transformer'] = results
        return results
    
    def compare_models(self) -> Dict:
        """
        Compare performance across all models.
        
        Returns:
            Dictionary of comparison results
        """
        if not self.results:
            raise ValueError("No models have been benchmarked yet.")
            
        comparison = {}
        
        # Compare metrics across models
        metrics = ['precision', 'recall', 'f1', 'accuracy',
                  'training_time', 'inference_time',
                  'training_memory', 'inference_memory']
        
        for metric in metrics:
            comparison[metric] = {
                model: results[metric]
                for model, results in self.results.items()
            }
        
        return comparison
    
    def plot_comparison(self) -> None:
        """
        Plot comparison of model performance.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if not self.results:
            raise ValueError("No models have been benchmarked yet.")
            
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # Plot performance metrics
        metrics = ['precision', 'recall', 'f1', 'accuracy']
        for i, metric in enumerate(metrics):
            values = [results[metric] for results in self.results.values()]
            models = list(self.results.keys())
            
            sns.barplot(x=models, y=values, ax=axes[i])
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylim(0, 1)
            
            # Add value labels
            for j, v in enumerate(values):
                axes[i].text(j, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Create subplots for computational metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # Plot computational metrics
        metrics = ['training_time', 'inference_time',
                  'training_memory', 'inference_memory']
        for i, metric in enumerate(metrics):
            values = [results[metric] for results in self.results.values()]
            models = list(self.results.keys())
            
            sns.barplot(x=models, y=values, ax=axes[i])
            axes[i].set_title(f'{metric.replace("_", " ").capitalize()} Comparison')
            
            # Add value labels
            for j, v in enumerate(values):
                axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate a detailed benchmark report.
        
        Returns:
            String containing the benchmark report
        """
        if not self.results:
            raise ValueError("No models have been benchmarked yet.")
            
        report = "Model Benchmarking Report\n"
        report += "=" * 50 + "\n\n"
        
        # Add results for each model
        for model, results in self.results.items():
            report += f"{model.upper()} Results:\n"
            report += "-" * 30 + "\n"
            
            # Performance metrics
            report += "Performance Metrics:\n"
            for metric in ['precision', 'recall', 'f1', 'accuracy']:
                report += f"{metric.capitalize()}: {results[metric]:.3f}\n"
            
            # Computational metrics
            report += "\nComputational Metrics:\n"
            report += f"Training Time: {results['training_time']:.2f} seconds\n"
            report += f"Inference Time: {results['inference_time']:.2f} seconds\n"
            report += f"Training Memory: {results['training_memory']:.2f} MB\n"
            report += f"Inference Memory: {results['inference_memory']:.2f} MB\n"
            
            report += "\n"
        
        return report 