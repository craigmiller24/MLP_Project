import pandas as pd
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
import torch.nn.functional as F  # For softmax

def plot_shap_values_DT(decision_tree_model, X_train_torch, X_test_torch, feature_names, location_code, save_path="shap_plot.png", sample_size=500):
    """
    Computes, plots, and saves SHAP values for a Decision Tree model using a sample of test data.
    
    Parameters:
    - decision_tree_model: Trained Decision Tree model object
    - X_train_torch: PyTorch tensor of training data
    - X_test_torch: PyTorch tensor of test data
    - feature_names: List of feature names after preprocessing
    - location_code: 3-letter airport/location code (e.g., 'jfk')
    - save_path: File path to save the figure (default: 'shap_plot.png')
    - sample_size: Number of samples to use for SHAP calculation (default: 500)
    
    Returns:
    - None (displays and saves SHAP summary plot)
    """
    
    # Convert PyTorch tensors to NumPy arrays
    X_train_np = X_train_torch.numpy()
    X_test_np = X_test_torch.numpy()

    # Randomly sample `sample_size` rows from X_test_np
    sample_size = min(sample_size, X_test_np.shape[0])  # Ensure we don’t sample more than available data
    sample_indices = np.random.choice(X_test_np.shape[0], sample_size, replace=False)
    X_test_sample = X_test_np[sample_indices]

    # Create SHAP explainer for the decision tree
    explainer = shap.TreeExplainer(decision_tree_model.model)

    # Compute SHAP values for the sampled test set
    shap_values = explainer.shap_values(X_test_sample)

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
    shap.summary_plot(filtered_shap_values, X_test_sample[:, important_features_mask], feature_names=filtered_feature_names, show=False)
    
    # Add title to the plot
    plt.title(f"SHAP Summary Plot for {location_code.upper()} (Sampled {sample_size} points)", fontsize=14)

    # Save the figure
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    print(f"SHAP summary plot generated and saved as '{save_path}' for {location_code.upper()} (Sampled {sample_size} points)")

def plot_shap_values_MLR(model, X_train_torch, X_test_torch, feature_names, location_code, save_path="shap_plot.png", sample_size=500):
    """
    Computes, plots, and saves SHAP values for a Multinomial Logistic Regression model using a sample of test data.

    Parameters:
    - model: Trained Multinomial Logistic Regression model object (with a .forward method)
    - X_train_torch: PyTorch tensor of training data
    - X_test_torch: PyTorch tensor of test data
    - feature_names: List of feature names after preprocessing
    - location_code: 3-letter airport/location code (e.g., 'jfk')
    - save_path: File path to save the figure (default: 'shap_plot.png')
    - sample_size: Number of samples to use for SHAP calculation (default: 500)

    Returns:
    - None (displays and saves SHAP summary plot)
    """
    
    # Convert PyTorch tensors to NumPy arrays
    X_train_np = X_train_torch.numpy()
    X_test_np = X_test_torch.numpy()

    # Randomly sample `sample_size` rows from X_test_np
    sample_size = min(sample_size, X_test_np.shape[0])  # Ensure we don’t sample more than available data
    sample_indices = np.random.choice(X_test_np.shape[0], sample_size, replace=False)
    X_test_sample = X_test_np[sample_indices]

    # Define a function to compute predictions from the MLR model
    def model_predict(x):
        # Ensure x is a PyTorch tensor
        x_tensor = torch.tensor(x, dtype=torch.float32)
        # Forward pass to get logits (raw outputs)
        logits = model(x_tensor)
        return logits.detach().numpy()

    # Create SHAP explainer using KernelExplainer
    explainer = shap.LinearExplainer(model_predict, X_train_np[:100])  # Using a subset of training data as background
    
    # Compute SHAP values for the sampled test set
    shap_values = explainer.shap_values(X_test_sample)

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
    shap.summary_plot(filtered_shap_values, X_test_sample[:, important_features_mask], feature_names=filtered_feature_names, show=False)
    
    # Add title to the plot
    plt.title(f"SHAP Summary Plot for {location_code.upper()} (Sampled {sample_size} points)", fontsize=14)

    # Save the figure
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    print(f"SHAP summary plot generated and saved as '{save_path}' for {location_code.upper()} (Sampled {sample_size} points)")

def plot_shap_values_RF(random_forest_model, X_train_torch, X_test_torch, feature_names, location_code, save_path="shap_plot_rf.png", sample_size=500):
    """
    Computes, plots, and saves SHAP values for a Random Forest model using a sample of test data.
    
    Parameters:
    - random_forest_model: Trained Random Forest model object
    - X_train_torch: PyTorch tensor of training data
    - X_test_torch: PyTorch tensor of test data
    - feature_names: List of feature names after preprocessing
    - location_code: 3-letter airport/location code (e.g., 'jfk')
    - save_path: File path to save the figure (default: 'shap_plot_rf.png')
    - sample_size: Number of samples to use for SHAP calculation (default: 500)
    
    Returns:
    - None (displays and saves SHAP summary plot)
    """
    
    # Convert PyTorch tensors to NumPy arrays
    X_train_np = X_train_torch.numpy()
    X_test_np = X_test_torch.numpy()

    # Randomly sample `sample_size` rows from X_test_np
    sample_size = min(sample_size, X_test_np.shape[0])  # Ensure we don't sample more than available data
    sample_indices = np.random.choice(X_test_np.shape[0], sample_size, replace=False)
    X_test_sample = X_test_np[sample_indices]

    # Create SHAP explainer for the random forest
    explainer = shap.TreeExplainer(random_forest_model.model)

    # Compute SHAP values for the sampled test set
    shap_values = explainer.shap_values(X_test_sample)

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
    shap.summary_plot(filtered_shap_values, X_test_sample[:, important_features_mask], feature_names=filtered_feature_names, show=False)
    
    # Add title to the plot
    plt.title(f"Random Forest SHAP Summary Plot for {location_code.upper()} (Sampled {sample_size} points)", fontsize=14)

    # Save the figure
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    print(f"Random Forest SHAP summary plot generated and saved as '{save_path}' for {location_code.upper()} (Sampled {sample_size} points)")

def plot_shap_values_LSTM(model, X_train_torch, X_test_torch, feature_names, location_code, save_path="shap_plot_lstm.png", sample_size=100, n_background=50):
    """
    Computes, plots, and saves SHAP values for an LSTM model using a sample of test data.

    Parameters:
    - model: Trained LSTM model object (with a .forward method)
    - X_train_torch: PyTorch tensor of training data
    - X_test_torch: PyTorch tensor of test data
    - feature_names: List of feature names after preprocessing
    - location_code: 3-letter airport/location code (e.g., 'jfk')
    - save_path: File path to save the figure (default: 'shap_plot_lstm.png')
    - sample_size: Number of samples to use for SHAP calculation (default: 100)
    - n_background: Number of background samples to use (default: 50)

    Returns:
    - None (displays and saves SHAP summary plot)
    """
    
    # Convert PyTorch tensors to NumPy arrays
    X_train_np = X_train_torch.numpy()
    X_test_np = X_test_torch.numpy()

    # Use fewer samples for LSTM due to computational complexity
    sample_size = min(sample_size, X_test_np.shape[0])
    sample_indices = np.random.choice(X_test_np.shape[0], sample_size, replace=False)
    X_test_sample = X_test_np[sample_indices]
    
    # Use fewer background samples for LSTM
    background_indices = np.random.choice(X_train_np.shape[0], n_background, replace=False)
    background_data = X_train_np[background_indices]
    
    # Set model to evaluation mode
    model.eval()

    # Define a prediction function that handles the LSTM's expected input format
    def model_predict(x):
        # Handle different input types and shapes
        if isinstance(x, list):
            x = np.array(x)
        
        # Create a PyTorch tensor
        x_tensor = torch.tensor(x, dtype=torch.float32)
        
        # Determine if we need to add sequence dimension based on your model's requirements
        # This part might need adjustment based on your specific LSTM architecture
        if len(x_tensor.shape) == 2:
            # If batch of features without sequence dimension
            batch_size, features = x_tensor.shape
            # Add sequence dimension (assuming sequence length of 1)
            x_tensor = x_tensor.view(batch_size, 1, features)
        elif len(x_tensor.shape) == 1:
            # If single sample
            x_tensor = x_tensor.view(1, 1, -1)
            
        # Add batch dimension if needed
        with torch.no_grad():
            try:
                output = model(x_tensor)
                # Convert output to numpy - handle different output formats
                if isinstance(output, tuple):
                    # Some LSTMs return (output, hidden_state)
                    output = output[0]
                
                # Get probabilities if needed
                if hasattr(model, 'softmax') and callable(getattr(model, 'softmax')):
                    output = model.softmax(output)
                
                return output.detach().numpy()
            except Exception as e:
                print(f"Error in prediction function: {e}")
                # Return a fallback value
                return np.zeros((x_tensor.shape[0], model.output_size))

    # Try both DeepExplainer and KernelExplainer depending on model complexity
    try:
        print("Attempting to use DeepExplainer for faster computation...")
        # Convert background data to tensor with proper shape
        background_tensor = torch.tensor(background_data, dtype=torch.float32)
        if len(background_tensor.shape) == 2:
            background_tensor = background_tensor.view(background_tensor.shape[0], 1, -1)
            
        # Try DeepExplainer first (faster but more restrictive)
        explainer = shap.DeepExplainer(model, background_tensor)
        # Get a small batch to test
        test_batch = torch.tensor(X_test_sample[:5], dtype=torch.float32)
        if len(test_batch.shape) == 2:
            test_batch = test_batch.view(test_batch.shape[0], 1, -1)
        # Test the explainer
        _ = explainer.shap_values(test_batch)
        # If it works, proceed with full calculation
        shap_tensor = torch.tensor(X_test_sample, dtype=torch.float32)
        if len(shap_tensor.shape) == 2:
            shap_tensor = shap_tensor.view(shap_tensor.shape[0], 1, -1)
        shap_values = explainer.shap_values(shap_tensor)
        deep_explainer_worked = True
    except Exception as e:
        print(f"DeepExplainer failed: {e}")
        print("Falling back to KernelExplainer (slower but more compatible)...")
        deep_explainer_worked = False
        # Fallback to KernelExplainer
        explainer = shap.KernelExplainer(model_predict, background_data)
        shap_values = explainer.shap_values(X_test_sample)
    
    # Process SHAP values based on their format
    if isinstance(shap_values, list):
        # Multi-class case
        print(f"Multi-class output detected with {len(shap_values)} classes")
        # Compute mean absolute SHAP values across classes
        shap_values_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        # Single output case
        print("Single output detected")
        shap_values_mean = np.abs(shap_values)
    
    # Remove sequence dimension if it exists
    if deep_explainer_worked and len(shap_values_mean.shape) > 2:
        shap_values_mean = shap_values_mean.squeeze(axis=1)
    
    # Compute feature importance and filter
    feature_importance = np.mean(shap_values_mean, axis=0)
    important_features_mask = feature_importance > 0
    
    # Check if any features are important
    if not np.any(important_features_mask):
        print("Warning: No features with importance > 0 found. Using all features.")
        important_features_mask = np.ones_like(feature_importance, dtype=bool)
    
    filtered_shap_values = shap_values_mean[:, important_features_mask]
    filtered_feature_names = np.array(feature_names)[important_features_mask]
    filtered_data = X_test_sample[:, important_features_mask]
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Create the SHAP summary plot
    if isinstance(shap_values, list) and not deep_explainer_worked:
        # For multi-class with KernelExplainer
        filtered_class_shap_values = [sv[:, important_features_mask] for sv in shap_values]
        shap.summary_plot(filtered_class_shap_values, filtered_data, 
                         feature_names=filtered_feature_names, show=False)
    else:
        # For single output or DeepExplainer results
        shap.summary_plot(filtered_shap_values, filtered_data, 
                         feature_names=filtered_feature_names, show=False)
    
    # Add title
    plt.title(f"LSTM SHAP Summary Plot for {location_code.upper()} (Sampled {sample_size} points)", fontsize=14)
    
    # Save the figure
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"LSTM SHAP summary plot generated and saved as '{save_path}' for {location_code.upper()}")
    print(f"Used {'DeepExplainer' if deep_explainer_worked else 'KernelExplainer'} with {sample_size} samples")

def plot_shap_values_RF_LSTM(lstm_model, X_train_torch, X_test_torch, feature_names, rf_model, location_code, save_path="shap_plot_rf_lstm.png", sample_size=100, top_n_features=10):
    """
    Computes, plots, and saves SHAP values for an RF-LSTM model where RF is used for feature selection.

    Parameters:
    - lstm_model: Trained LSTM model that uses RF-selected features
    - X_train_torch: Original PyTorch tensor of training data
    - X_test_torch: Original PyTorch tensor of test data
    - feature_names: List of all feature names
    - rf_model: The trained Random Forest model used for feature selection
    - location_code: 3-letter airport/location code (e.g., 'jfk')
    - save_path: File path to save the figure (default: 'shap_plot_rf_lstm.png')
    - sample_size: Number of samples to use for SHAP calculation (default: 100)
    - top_n_features: Number of top features selected by RF (default: 10)

    Returns:
    - None (displays and saves SHAP summary plot)
    """
    
    # Get feature importances from RF
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': rf_model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    # Select top N important features
    top_features = feature_importance.head(top_n_features)['Feature'].tolist()
    top_feature_indices = [feature_names.index(feat) for feat in top_features]
    
    # Get the reduced dataset with only selected features
    X_train_selected = X_train_torch[:, top_feature_indices].numpy()
    X_test_selected = X_test_torch[:, top_feature_indices].numpy()
    
    # Use fewer samples for computational efficiency
    sample_size = min(sample_size, X_test_selected.shape[0])
    sample_indices = np.random.choice(X_test_selected.shape[0], sample_size, replace=False)
    X_test_sample = X_test_selected[sample_indices]
    
    # Use a smaller background dataset
    n_background = min(50, X_train_selected.shape[0])
    background_indices = np.random.choice(X_train_selected.shape[0], n_background, replace=False)
    background_data = X_train_selected[background_indices]
    
    # Set model to evaluation mode
    lstm_model.eval()
    
    # Define prediction function for LSTM model
    def lstm_predict(x):
        # Convert to PyTorch tensor
        x_tensor = torch.tensor(x, dtype=torch.float32)
        
        with torch.no_grad():
            try:
                # Forward pass through LSTM
                outputs = lstm_model(x_tensor)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                return outputs.detach().numpy()
            except Exception as e:
                print(f"Error in LSTM prediction: {e}")
                # Return fallback value based on your model's output size
                return np.zeros((x.shape[0], lstm_model.fc.out_features))
    
    # Try both explanation methods
    try:
        print("Attempting to use DeepExplainer...")
        background_tensor = torch.tensor(background_data, dtype=torch.float32)
        explainer = shap.DeepExplainer(lstm_model, background_tensor)
        
        # Test on small batch
        test_batch = torch.tensor(X_test_sample[:5], dtype=torch.float32)
        _ = explainer.shap_values(test_batch)
        
        # If successful, compute for all samples
        shap_tensor = torch.tensor(X_test_sample, dtype=torch.float32)
        shap_values = explainer.shap_values(shap_tensor)
        deep_explainer_worked = True
    except Exception as e:
        print(f"DeepExplainer failed: {e}")
        print("Falling back to KernelExplainer...")
        deep_explainer_worked = False
        explainer = shap.KernelExplainer(lstm_predict, background_data)
        shap_values = explainer.shap_values(X_test_sample)
    
    # Process SHAP values
    if isinstance(shap_values, list):
        print(f"Multi-class output detected with {len(shap_values)} classes")
        shap_values_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        print("Single output detected")
        shap_values_mean = np.abs(shap_values)
    
    # Map SHAP values back to original feature names
    # We need to create a mapping from the selected features to their names
    selected_feature_names = [feature_names[i] for i in top_feature_indices]
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Create the SHAP summary plot
    if isinstance(shap_values, list) and not deep_explainer_worked:
        # For multi-class with KernelExplainer
        shap.summary_plot(shap_values, X_test_sample, feature_names=selected_feature_names, show=False)
    else:
        # For single output or DeepExplainer results
        shap.summary_plot(shap_values_mean, X_test_sample, feature_names=selected_feature_names, show=False)
    
    # Add title with RF feature selection note
    plt.title(f"RF-LSTM SHAP Plot for {location_code.upper()} (Top {top_n_features} RF-selected features)", fontsize=14)
    
    # Save the figure
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"RF-LSTM SHAP plot generated and saved as '{save_path}' for {location_code.upper()}")
    print(f"Using {top_n_features} features selected by Random Forest")