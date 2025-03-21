# preprocessing gpu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch

# Set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def preprocess_data_gpu(df, test_size=0.2, random_state=42):
    # Define target variables and feature variables
    X = df.drop(columns=['DEP_DATE_TIME', 'ACC_DEP_TIME', 'DEP_DELAY_NEW', 'DEP_DELAY_GROUP', 'WEATHER_DELAY', 'CANCELLED', 'NAS_DELAY'])
    y = df['DEP_DELAY_GROUP'].replace({-2: 0, -1: 0})  # Replace -2 and -1 with 0
    
    # Define numerical and categorical columns
    categorical_cols = ['ORIGIN', 'DEST']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    # Train-test split (not random, but sequential)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Ensure encoder learns all categories, even those missing in training
    all_categories = {col: X[col].unique() for col in categorical_cols}
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(categories=[all_categories[col] for col in categorical_cols], drop='first', handle_unknown='ignore'), categorical_cols)
        ])
    
    # Preprocess the data (Fit on training data and transform both training and test data)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_torch = torch.FloatTensor(X_train_processed.toarray())
    Y_train_torch = torch.LongTensor(Y_train.values)
    X_test_torch = torch.FloatTensor(X_test_processed.toarray())
    Y_test_torch = torch.LongTensor(Y_test.values)
    
    # Generate feature names after OneHotEncoding for categorical columns
    categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(categorical_feature_names)
    
    # Move tensors to device (GPU if available, CPU otherwise)
    X_train_torch = X_train_torch.to(device)
    Y_train_torch = Y_train_torch.to(device)
    X_test_torch = X_test_torch.to(device)
    Y_test_torch = Y_test_torch.to(device)
    
    return X_train_torch, X_test_torch, Y_train_torch, Y_test_torch, feature_names