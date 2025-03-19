import pandas as pd
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
import torch.nn.functional as F  # For softmax
from tqdm import tqdm  # Import tqdm for progress bar

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
    sample_size = min(sample_size, X_test_np.shape[0])
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
    sample_size = min(sample_size, X_test_np.shape[0]) 
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
    sample_size = min(sample_size, X_test_np.shape[0])
    sample_indices = np.random.choice(X_test_np.shape[0], sample_size, replace=False)
    X_test_sample = X_test_np[sample_indices]

    # Create SHAP explainer for the random forest
    explainer = shap.TreeExplainer(random_forest_model)

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

def plot_shap_values_LSTM(model, X_train_torch, X_test_torch, feature_names, location_code, save_path="shap_plot.png", sample_size=500, background_size=50):
    """
    Computes, plots, and saves SHAP values for an LSTM model using a sample of test data.

    Parameters:
    - model: Trained LSTM model object (with a .forward method)
    - X_train_torch: PyTorch tensor of training data
    - X_test_torch: PyTorch tensor of test data
    - feature_names: List of feature names after preprocessing
    - location_code: 3-letter airport/location code (e.g., 'jfk')
    - save_path: File path to save the figure (default: 'shap_plot.png')
    - sample_size: Number of samples to use for SHAP calculation (default: 500)
    - background_size: Number of background samples for SHAP (default: 50)

    Returns:
    - None (displays and saves SHAP summary plot)
    """

    # Convert PyTorch tensors to NumPy arrays
    if isinstance(X_train_torch, torch.Tensor):
        X_train_np = X_train_torch.numpy()  # Convert to NumPy array if it's a tensor
    else:
        X_train_np = X_train_torch  # Already a NumPy array

    if isinstance(X_test_torch, torch.Tensor):
        X_test_np = X_test_torch.numpy()  # Convert to NumPy array if it's a tensor
    else:
        X_test_np = X_test_torch  # Already a NumPy array

    # Reduce the number of background samples using shap.kmeans (e.g., 1000 samples)
    X_train_summarized = shap.kmeans(X_train_np, background_size)  # Summarizing training data to `background_size` samples

    # Select a random sample (for example, 50 samples)
    sample_size = min(sample_size, X_test_np.shape[0]) 
    random_indices = np.random.choice(X_test_np.shape[0], size=sample_size, replace=False)

    # Extract the sample data
    X_sample = X_test_np[random_indices]

    # Define a prediction function for SHAP that works with your PyTorch model
    def predict_fn(X):
        # Convert the NumPy input to a PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        # Get the model's predictions
        with torch.no_grad():  # Disable gradient calculation for inference
            predictions = model(X_tensor)  # Get raw output from the LSTM model
        return predictions.numpy()

    # Create a SHAP explainer
    explainer = shap.KernelExplainer(predict_fn, X_train_summarized)

    # Calculate SHAP values for the sample with a progress bar
    shap_values_sample = []
    with tqdm(total=sample_size, desc="Computing SHAP values", unit="sample") as pbar:
        for i in range(sample_size):
            shap_values_sample.append(explainer.shap_values(X_sample[i:i+1]))  # Process one sample at a time
            pbar.update(1)  # Update the progress bar

    # Convert the list of SHAP values to a NumPy array
    shap_values_sample = np.array(shap_values_sample)

    # Check the shape of shap_values_sample and X_sample
    print(f"shap_values_sample shape: {shap_values_sample.shape}")
    print(f"X_sample shape: {X_sample.shape}")

    # Nreshape shap_values_sample to match X_sample (50, 102)
    # Remove extra dimensions (class dimension)
    if shap_values_sample.ndim == 4:
        # Remove the middle dimension
        shap_values_sample = shap_values_sample[:, 0, :, :]  # Shape: (50, 102, 13)
        shap_values_sample = shap_values_sample[:, :, 0]
        # shap_values_sample should have shape: (50, 102)

    # Visualize the SHAP values for the sample
    plt.figure(figsize=(12, 8))
    
    # Use the matplotlib figure directly
    fig = shap.summary_plot(
        shap_values_sample, 
        X_sample, 
        feature_names=feature_names,
        show=False
    )
    
    # Add title to the plot
    plt.title(f"SHAP Summary Plot for {location_code.upper()} (Sampled {sample_size} points)", fontsize=14)
    
    # Save the figure
    plt.tight_layout()  # Adjust the layout
    plt.savefig(save_path, bbox_inches="tight", dpi=300)  # Increase dpi for better quality
    
    # Close the figure to free memory
    plt.close()
    
    print(f"SHAP summary plot generated and saved as '{save_path}' for {location_code.upper()} (Sampled {sample_size} points)")

def plot_shap_values_RF_LSTM(model, X_train_torch, X_test_torch, feature_names, location_code, 
                              save_path="RFLSTM_shap_plot.png", sample_size=500, background_size=50):
    """
    Computes, plots, and saves SHAP values for an ensemble model using a sample of test data.

    Parameters:
    - model: Trained ensemble model object (with a .forward method)
    - X_train_torch: PyTorch tensor of training data
    - X_test_torch: PyTorch tensor of test data
    - feature_names: List of feature names after preprocessing
    - location_code: 3-letter airport/location code (e.g., 'jfk')
    - save_path: File path to save the figure (default: 'ensemble_shap_plot.png')
    - sample_size: Number of samples to use for SHAP calculation (default: 500)
    - background_size: Number of background samples for SHAP (default: 50)

    Returns:
    - None (displays and saves SHAP summary plot)
    """
    # Convert PyTorch tensors to NumPy arrays
    if isinstance(X_train_torch, torch.Tensor):
        X_train_np = X_train_torch.numpy()  # Convert to NumPy array if it's a tensor
    else:
        X_train_np = X_train_torch  # Already a NumPy array

    if isinstance(X_test_torch, torch.Tensor):
        X_test_np = X_test_torch.numpy()  # Convert to NumPy array if it's a tensor
    else:
        X_test_np = X_test_torch  # Already a NumPy array

    # Reduce the number of background samples using shap.kmeans
    X_train_summarized = shap.kmeans(X_train_np, background_size)  # Summarizing training data to `background_size` samples

    # Select a random sample
    sample_size = min(sample_size, X_test_np.shape[0])  # Ensure we don't sample more than available data
    random_indices = np.random.choice(X_test_np.shape[0], size=sample_size, replace=False)

    # Extract the sample data
    X_sample = X_test_np[random_indices]

    # Define a prediction function for SHAP that works with your ensemble model
    def predict_fn(X):
        # Convert the NumPy input to a PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Set model to evaluation mode
        model.eval()
        
        # Get the model's predictions
        with torch.no_grad():  # Disable gradient calculation for inference
            predictions = model(X_tensor)  # Get raw output from the model
        return predictions.numpy()

    # Create a SHAP explainer (using KernelExplainer for black-box models)
    explainer = shap.KernelExplainer(predict_fn, X_train_summarized)

    # Calculate SHAP values for the sample with a progress bar using tqdm
    shap_values_sample = []
    with tqdm(total=sample_size, desc="Computing SHAP values", unit="sample") as pbar:
        for i in range(sample_size):
            shap_values_sample.append(explainer.shap_values(X_sample[i:i+1]))  # Process one sample at a time
            pbar.update(1)  # Update the progress bar

    # Convert the list of SHAP values to a NumPy array (flatten to the right shape)
    shap_values_sample = np.array(shap_values_sample)

    # Check the shape of shap_values_sample and X_sample
    print(f"shap_values_sample shape: {shap_values_sample.shape}")
    print(f"X_sample shape: {X_sample.shape}")

    # Now let's reshape shap_values_sample to match X_sample
    # Remove extra dimensions (e.g., class dimension)
    if shap_values_sample.ndim == 4:
        # Remove the middle dimension (this assumes it's for classes)
        shap_values_sample = shap_values_sample[:, 0, :, :]
        shap_values_sample = shap_values_sample[:, :, 0]

    # Visualize the SHAP values for the sample
    plt.figure(figsize=(12, 8))  # Explicitly create a figure with a specific size
    
    # Use the matplotlib figure directly
    fig = shap.summary_plot(
        shap_values_sample, 
        X_sample, 
        feature_names=feature_names,
        show=False  # Important: don't show the plot yet
    )
    
    # Add title to the plot
    plt.title(f"SHAP Summary Plot for {location_code.upper()} Ensemble Model\n(Sampled {sample_size} points)", fontsize=14)
    
    # Save the figure
    plt.tight_layout()  # Adjust the layout
    plt.savefig(save_path, bbox_inches="tight", dpi=300)  # Increase dpi for better quality
    
    # Show the plot if needed (you can comment this out if you only want to save)
    plt.show()
    
    # Close the figure to free memory
    plt.close()
    
    print(f"RF-LSTM SHAP summary plot generated and saved as '{save_path}' for {location_code.upper()} (Sampled {sample_size} points)")