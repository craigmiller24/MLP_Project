import pandas as pd
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import gc

# Device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_shap_values_DT(decision_tree_model, X_train_torch, X_test_torch, feature_names, location_code, save_path="shap_plot.png"):
    """
    Computes, plots, and saves SHAP values for a Decision Tree model.
    Works with data on either GPU or CPU.
    
    Parameters:
    - decision_tree_model: Trained Decision Tree model object
    - X_train_torch: PyTorch tensor of training data (can be on GPU or CPU)
    - X_test_torch: PyTorch tensor of test data (can be on GPU or CPU)
    - feature_names: List of feature names after preprocessing
    - location_code: 3-letter airport/location code (e.g., 'jfk')
    - save_path: File path to save the figure (default: 'shap_plot.png')
    
    Returns:
    - None (displays and saves SHAP summary plot)
    """
    
    # Check if X_train_torch and X_test_torch are lists, then concatenate them if needed
    if isinstance(X_train_torch, list):
        X_train_torch = torch.cat(X_train_torch, dim=0)
    
    if isinstance(X_test_torch, list):
        X_test_torch = torch.cat(X_test_torch, dim=0)
    
    # Make sure tensors are on CPU and convert to NumPy arrays
    X_train_np = X_train_torch.cpu().numpy()
    X_test_np = X_test_torch.cpu().numpy()

    # Create SHAP explainer for the decision tree
    explainer = shap.TreeExplainer(decision_tree_model.model)

    # Compute SHAP values for test set
    shap_values = explainer.shap_values(X_test_np)

    # Compute mean absolute SHAP values across classes
    shap_values_mean = np.mean(np.abs(shap_values), axis=2)  # Shape: (num_samples, num_features)

    # Compute global feature importance by averaging over all samples
    feature_importance = np.mean(shap_values_mean, axis=0)  # Shape: (num_features,)

    # Filter features where importance > 0
    important_features_mask = feature_importance > 0
    filtered_shap_values = shap_values_mean[:, important_features_mask]
    filtered_feature_names = np.array(feature_names)[important_features_mask]

    # Create the figure
    plt.figure()
    shap.summary_plot(filtered_shap_values, X_test_np[:, important_features_mask], feature_names=filtered_feature_names, show=False)
    
    # Add title to the plot
    plt.title(f"SHAP Summary Plot for {location_code.upper()}", fontsize=14)

    # Save the figure
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    print(f"SHAP summary plot generated and saved as '{save_path}' for {location_code.upper()}")


def plot_shap_values_MLR(MLR_model, X_train_torch, X_test_torch, feature_names, location_code, save_path="shap_plot.png"):
    """
    Computes, plots, and saves SHAP values for a Multinomial Logistic Regression model.
    Works with data on either GPU or CPU.
    
    Parameters:
    - logistic_regression_model: Trained Logistic Regression model (e.g., sklearn or PyTorch model)
    - X_train_torch: PyTorch tensor of training data (can be on GPU or CPU)
    - X_test_torch: PyTorch tensor of test data (can be on GPU or CPU)
    - feature_names: List of feature names after preprocessing
    - location_code: 3-letter airport/location code (e.g., 'jfk')
    - save_path: File path to save the figure (default: 'shap_plot.png')
    
    Returns:
    - None (displays and saves SHAP summary plot)
    """
    # Make sure the model is in evaluation mode
    MLR_model.eval()
    
    # Create a PyTorch-compatible wrapper for the linear model
    class LinearModelWrapper:
        def __init__(self, model):
            self.model = model
            # Extract weights and bias from the PyTorch model's linear layer
            # Make sure to move to CPU for compatibility with SHAP
            self.weights = model.linear.weight.detach().cpu().numpy()
            self.bias = model.linear.bias.detach().cpu().numpy()
        
        def __call__(self, X):
            return X @ self.weights.T + self.bias

    # Wrap the MLR model
    wrapped_model = LinearModelWrapper(MLR_model)

    # Convert PyTorch tensors to NumPy arrays (make sure they're on CPU first)
    X_train_np = X_train_torch.cpu().numpy()
    X_test_np = X_test_torch.cpu().numpy()

    # Use SHAP's LinearExplainer for logistic regression
    explainer = shap.LinearExplainer(wrapped_model, X_train_np)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_test_np)  # Shape: (num_samples, num_features) per class

    # If it's multinomial logistic regression, shap_values is a list (one array per class)
    if isinstance(shap_values, list):
        shap_values = np.mean(np.abs(np.array(shap_values)), axis=0)  # Mean over classes
    
    # Compute global feature importance
    feature_importance = np.mean(np.abs(shap_values), axis=0)  # Shape: (num_features,)

    # Filter features where importance > 0
    important_features_mask = feature_importance > 0
    filtered_shap_values = shap_values[:, important_features_mask]
    filtered_feature_names = np.array(feature_names)[important_features_mask]

    # Create the figure
    plt.figure()
    shap.summary_plot(filtered_shap_values, X_test_np[:, important_features_mask], feature_names=filtered_feature_names, show=False)
    
    # Add title to the plot
    plt.title(f"SHAP Summary Plot for {location_code.upper()}", fontsize=14)

    # Save the figure
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    print(f"SHAP summary plot generated and saved as '{save_path}' for {location_code.upper()}")