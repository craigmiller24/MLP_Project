# import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

# Random Forest

def train_random_forest(X_train_torch, X_test_torch, Y_train_torch, Y_test_torch, feature_names):

    # Convert PyTorch tensors to NumPy arrays
    X_train_np = X_train_torch.numpy()
    X_test_np = X_test_torch.numpy()
    Y_train_np = Y_train_torch.numpy()
    Y_test_np = Y_test_torch.numpy()

    # Initialize Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=10,       # Max depth for each tree
        min_samples_leaf=50,  # Minimum samples per leaf
        min_samples_split=50,  # Minimum samples to split a node
        max_leaf_nodes=12,  # Max leaf nodes for each tree
        random_state=42
    )
    
    # Train the model
    rf_model.fit(X_train_np, Y_train_np)
    
    # Make predictions
    y_pred_train = rf_model.predict(X_train_np)
    y_pred_test = rf_model.predict(X_test_np)

    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(Y_train_np, y_pred_train),
        'train_precision': precision_score(Y_train_np, y_pred_train, average='weighted'),
        'train_recall': recall_score(Y_train_np, y_pred_train, average='weighted'),
        'train_f1': f1_score(Y_train_np, y_pred_train, average='weighted'),
        'test_accuracy': accuracy_score(Y_test_np, y_pred_test),
        'test_precision': precision_score(Y_test_np, y_pred_test, average='weighted'),
        'test_recall': recall_score(Y_test_np, y_pred_test, average='weighted'),
        'test_f1': f1_score(Y_test_np, y_pred_test, average='weighted')
    }

    # Print metrics
    print("\nTraining Set Metrics:")
    for key in metrics:
        if 'train' in key:
            print(f"{key.replace('_', ' ').title()}: {metrics[key]:.4f}")

    print("\nTest Set Metrics:")
    for key in metrics:
        if 'test' in key:
            print(f"{key.replace('_', ' ').title()}: {metrics[key]:.4f}")

    # Feature importance plot
    importance_data = pd.DataFrame({'Feature': feature_names, 'Importance': rf_model.feature_importances_})
    importance_data = importance_data[importance_data['Importance'] > 0].sort_values('Importance', ascending=True)

    plt.figure(figsize=(12, 6))
    plt.barh(importance_data['Feature'], importance_data['Importance'], color='skyblue', edgecolor='navy')
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importance in Random Forest', fontsize=14, pad=20)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig('Images/rf_feature_importance_plot.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    return rf_model, metrics


# LSTM

def train_lstm(X_train_torch, X_test_torch, Y_train_torch, Y_test_torch, hidden_size=64, num_epochs=100, batch_size=64, lr=0.001):
    # Define LSTM model class
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  # LSTM layer
            self.fc = nn.Linear(hidden_size, num_classes)  # Fully connected layer

        def forward(self, x):
            x = x.unsqueeze(1)  # Reshape input to (batch_size, 1, input_size)
            out, _ = self.lstm(x)  # LSTM forward pass
            out = out[:, -1, :]  # Get the last time step's output
            out = self.fc(out)  # Fully connected layer
            return out

    # Model hyperparameters
    input_size = X_train_torch.shape[1]  
    num_classes = len(torch.unique(Y_train_torch))  

    # Initialize model
    model = LSTMModel(input_size, hidden_size, num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # DataLoader setup
    train_data = TensorDataset(X_train_torch, Y_train_torch)
    test_data = TensorDataset(X_test_torch, Y_test_torch)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Tracking loss and accuracy
    train_loss_list = []
    train_accuracy_list = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)  
            loss = criterion(outputs, batch_y)  
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        # Average loss and accuracy for the epoch
        avg_loss = epoch_loss / len(train_loader)
        avg_accuracy = correct / total
        train_loss_list.append(avg_loss)
        train_accuracy_list.append(avg_accuracy)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}')

    # Evaluate on train set
    model.eval()
    y_pred_train = []
    y_true_train = []
    
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            y_pred_train.extend(predicted.cpu().numpy())
            y_true_train.extend(batch_y.cpu().numpy())
    
    # Evaluate on test set
    y_pred_test = []
    y_true_test = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            y_pred_test.extend(predicted.cpu().numpy())
            y_true_test.extend(batch_y.cpu().numpy())

    # Store metrics in a dictionary
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_true_train, y_pred_train),
        'train_precision': precision_score(y_true_train, y_pred_train, average='weighted'),
        'train_recall': recall_score(y_true_train, y_pred_train, average='weighted'),
        'train_f1': f1_score(y_true_train, y_pred_train, average='weighted'),
        'test_accuracy': accuracy_score(y_true_test, y_pred_test),
        'test_precision': precision_score(y_true_test, y_pred_test, average='weighted'),
        'test_recall': recall_score(y_true_test, y_pred_test, average='weighted'),
        'test_f1': f1_score(y_true_test, y_pred_test, average='weighted')
    }

    # Print evaluation metrics
    print(f'\nTraining Set Metrics:')
    print(f'Accuracy: {metrics["train_accuracy"]:.4f}')
    print(f'Precision: {metrics["train_precision"]:.4f}')
    print(f'Recall: {metrics["train_recall"]:.4f}')
    print(f'F1-score: {metrics["train_f1"]:.4f}')

    print(f'\nTest Set Metrics:')
    print(f'Accuracy: {metrics["test_accuracy"]:.4f}')
    print(f'Precision: {metrics["test_precision"]:.4f}')
    print(f'Recall: {metrics["test_recall"]:.4f}')
    print(f'F1-score: {metrics["test_f1"]:.4f}')

    # Plot loss
    plt.figure(figsize=(4, 4))
    plt.plot(range(1, num_epochs + 1), train_loss_list, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Across Epochs')
    plt.legend()
    plt.savefig('Images/training_loss_plot_LSTM.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(4, 4))
    plt.plot(range(1, num_epochs + 1), train_accuracy_list, label='Training Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Across Epochs')
    plt.legend()
    plt.savefig('Images/training_accuracy_plot_LSTM.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Return the model and the evaluation metrics
    return model, metrics


# Ensemble model - RF + LSTM

def train_rf_then_lstm(X_train_torch, X_test_torch, Y_train_torch, Y_test_torch, feature_names, 
                        top_n_features=10, hidden_size=64, num_epochs=100, batch_size=64, lr=0.001):

    # Convert PyTorch tensors to NumPy arrays for RF
    X_train_np = X_train_torch.numpy()
    X_test_np = X_test_torch.numpy()
    Y_train_np = Y_train_torch.numpy()
    Y_test_np = Y_test_torch.numpy()

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_np, Y_train_np)

    # Get feature importances
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': rf_model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # Select top N important features
    top_features = feature_importance.head(top_n_features)['Feature'].tolist()
    top_feature_indices = [feature_names.index(feat) for feat in top_features]

    # Reduce dataset to selected features
    X_train_selected = X_train_torch[:, top_feature_indices]
    X_test_selected = X_test_torch[:, top_feature_indices]

    # Define LSTM Model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = x.unsqueeze(1)  # Reshape input to (batch_size, 1, input_size)
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    # Initialize model
    input_size = X_train_selected.shape[1]
    num_classes = len(torch.unique(Y_train_torch))
    model = LSTMModel(input_size, hidden_size, num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # DataLoader setup
    train_data = TensorDataset(X_train_selected, Y_train_torch)
    test_data = TensorDataset(X_test_selected, Y_test_torch)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Training loop
    train_loss_list = []
    train_accuracy_list = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = epoch_loss / len(train_loader)
        avg_accuracy = correct / total
        train_loss_list.append(avg_loss)
        train_accuracy_list.append(avg_accuracy)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}')

    # Evaluate on train and test sets
    model.eval()
    y_pred_train = []
    y_true_train = []
    y_pred_test = []
    y_true_test = []

    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            y_pred_train.extend(predicted.cpu().numpy())
            y_true_train.extend(batch_y.cpu().numpy())

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            y_pred_test.extend(predicted.cpu().numpy())
            y_true_test.extend(batch_y.cpu().numpy())

    # Calculate train metrics
    metrics = {
        'train_accuracy': accuracy_score(y_true_train, y_pred_train),
        'train_precision': precision_score(y_true_train, y_pred_train, average='weighted'),
        'train_recall': recall_score(y_true_train, y_pred_train, average='weighted'),
        'train_f1_score': f1_score(y_true_train, y_pred_train, average='weighted'),
        'test_accuracy': accuracy_score(y_true_test, y_pred_test),
        'test_precision': precision_score(y_true_test, y_pred_test, average='weighted'),
        'test_recall': recall_score(y_true_test, y_pred_test, average='weighted'),
        'test_f1_score': f1_score(y_true_test, y_pred_test, average='weighted')
    }

    # Print evaluation metrics
    print(f'\nTraining Set Metrics:')
    print(f'Accuracy: {metrics["train_accuracy"]:.4f}')
    print(f'Precision: {metrics["train_precision"]:.4f}')
    print(f'Recall: {metrics["train_recall"]:.4f}')
    print(f'F1-score: {metrics["train_f1_score"]:.4f}')

    print(f'\nTest Set Metrics:')
    print(f'Accuracy: {metrics["test_accuracy"]:.4f}')
    print(f'Precision: {metrics["test_precision"]:.4f}')
    print(f'Recall: {metrics["test_recall"]:.4f}')
    print(f'F1-score: {metrics["test_f1_score"]:.4f}')

    # Plot accuracy and loss
    plt.figure(figsize=(4, 4))
    plt.plot(range(1, num_epochs + 1), train_loss_list, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Across Epochs')
    plt.legend()
    plt.savefig('Images/training_loss_plot_LSTM.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(4, 4))
    plt.plot(range(1, num_epochs + 1), train_accuracy_list, label='Training Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Across Epochs')
    plt.legend()
    plt.savefig('Images/training_accuracy_plot_LSTM.png', dpi=300, bbox_inches='tight')
    plt.show()

    return model, metrics, top_features