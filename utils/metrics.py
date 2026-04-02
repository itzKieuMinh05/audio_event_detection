"""
Metrics Calculator for Audio Event Detection
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score
)
from typing import Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """
    Calculate and track evaluation metrics
    """
    
    def __init__(self, num_classes: int, class_names: Optional[list] = None):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of classes
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
    
    def calculate_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # Advanced metrics (if probabilities provided)
        if y_prob is not None:
            try:
                # ROC AUC (one-vs-rest)
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_prob, 
                    multi_class='ovr', 
                    average='macro',
                    labels=np.arange(self.num_classes)
                )
                
                # Mean Average Precision
                metrics['map'] = average_precision_score(
                    self._to_one_hot(y_true),
                    y_prob,
                    average='macro'
                )
            except Exception as e:
                print(f"Warning: Could not calculate advanced metrics: {e}")
        
        return metrics
    
    def _to_one_hot(self, labels: np.ndarray) -> np.ndarray:
        """Convert labels to one-hot encoding"""
        one_hot = np.zeros((len(labels), self.num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             save_path: Optional[str] = None,
                             normalize: bool = True):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot
            normalize: Whether to normalize
        """
        cm = self.get_confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Get detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        return classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0
        )
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Print metrics in a formatted way
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*60)
        print("Evaluation Metrics")
        print("="*60)
        
        # Overall metrics
        print("\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        if 'map' in metrics:
            print(f"  mAP:       {metrics['map']:.4f}")
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        for class_name in self.class_names:
            if f'f1_{class_name}' in metrics:
                print(f"  {class_name}:")
                print(f"    Precision: {metrics[f'precision_{class_name}']:.4f}")
                print(f"    Recall:    {metrics[f'recall_{class_name}']:.4f}")
                print(f"    F1-Score:  {metrics[f'f1_{class_name}']:.4f}")
        
        print("="*60)


def test_metrics():
    """Test metrics calculator"""
    print("Testing metrics calculator...")
    
    # Create dummy data
    num_classes = 7
    class_names = ['gunshot', 'explosion', 'siren', 'glass_breaking', 
                   'scream', 'dog_bark', 'fire_crackling']
    
    y_true = np.random.randint(0, num_classes, 100)
    y_pred = np.random.randint(0, num_classes, 100)
    y_prob = np.random.rand(100, num_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # Calculate metrics
    calculator = MetricsCalculator(num_classes, class_names)
    metrics = calculator.calculate_metrics(y_true, y_pred, y_prob)
    
    # Print metrics
    calculator.print_metrics(metrics)
    
    # Print classification report
    print("\n" + calculator.get_classification_report(y_true, y_pred))
    
    print("\nMetrics test complete!")


if __name__ == "__main__":
    test_metrics()
