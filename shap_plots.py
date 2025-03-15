import pandas as pd
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import shap

def plot_shap_values_DT(decision_tree_model, X_train_torch, X_test_torch, feature_names, location_code, save_path="shap_plot.png"):
    """
    Computes, plots, and saves SHAP values for a Decision Tree model.
    
    Parameters:
    - decision_tree_model: Trained Decision Tree model object
    - X_train_torch: PyTorch tensor of training data
    - X_test_torch: PyTorch tensor of test data
    - feature_names: List of feature names after preprocessing
    - location_code: 3-letter airport/location code (e.g., 'jfk')
    - save_path: File path to save the figure (default: 'shap_plot.png')
    
    Returns:
    - None (displays and saves SHAP summary plot)
    """
    
    # Convert PyTorch tensors to NumPy arrays
    X_train_np = X_train_torch.numpy()
    X_test_np = X_test_torch.numpy()

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
    Computes, plots, and saves SHAP values for a Multinomial Logistic Regression model in PyTorch.
    
    Parameters:
    - MLR_model: Trained PyTorch Logistic Regression model
    - X_train_torch: PyTorch tensor of training data
    - X_test_torch: PyTorch tensor of test data
    - feature_names: List of feature names after preprocessing
    - location_code: 3-letter airport/location code (e.g., 'jfk')
    - save_path: File path to save the figure (default: 'shap_plot.png')
    
    Returns:
    - None (displays and saves SHAP summary plot)
    """
    # Extract weights from PyTorch model
    weights = MLR_model.linear.weight.detach().numpy()
    bias = MLR_model.linear.bias.detach().numpy()
    
    # Convert PyTorch tensors to NumPy arrays
    X_train_np = X_train_torch.detach().numpy()
    X_test_np = X_test_torch.detach().numpy()

    # Create a Kernel explainer since we're working with a custom model
    # First create a function that returns model predictions
    def model_predict(X):
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            output = MLR_model(X_tensor)
            # Convert to probabilities using softmax
            probs = torch.nn.functional.softmax(output, dim=1).numpy()
        return probs
    
    # Use a subset of training data as background for the explainer
    background = shap.kmeans(X_train_np, 100)  # Use KMeans to summarize the training data
    
    # Create explainer
    explainer = shap.KernelExplainer(model_predict, background)
    
    # Compute SHAP values on a subset of test data for efficiency
    sample_size = min(500, X_test_np.shape[0])  # Use at most 500 test samples
    X_test_sample = X_test_np[:sample_size]
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_test_sample)  # This returns a list of arrays (one per class)
    
    # For multiclass, take the mean absolute SHAP value across classes
    if isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values)
    
    # Compute feature importance
    feature_importance = np.mean(mean_shap, axis=0)
    
    # Filter features where importance > 0
    important_idx = np.where(feature_importance > 0)[0]
    if len(important_idx) == 0:  # If no features meet the criteria, use all features
        important_idx = np.arange(len(feature_names))
    
    # Get filtered features and SHAP values
    filtered_features = [feature_names[i] for i in important_idx]
    filtered_X_test = X_test_sample[:, important_idx]
    
    if isinstance(shap_values, list):
        filtered_shap_values = [sv[:, important_idx] for sv in shap_values]
    else:
        filtered_shap_values = shap_values[:, important_idx]
    
    # Create and save the plot
    plt.figure(figsize=(12, 8))
    if isinstance(filtered_shap_values, list):
        # For multiclass, plot the mean absolute SHAP values
        shap.summary_plot(filtered_shap_values[0], filtered_X_test, 
                         feature_names=filtered_features, show=False)
    else:
        shap.summary_plot(filtered_shap_values, filtered_X_test, 
                         feature_names=filtered_features, show=False)
    
    plt.title(f"SHAP Summary Plot for {location_code.upper()}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    print(f"SHAP summary plot generated and saved as '{save_path}' for {location_code.upper()}")