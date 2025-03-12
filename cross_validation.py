# General function for 5-fold cross-validation
import numpy as np
import pandas as pd
from DataPreprocessing.preprocessing_function import preprocess_no_split
import inspect

def crossvalidation_training(model_name, airports, airport_data_dict):
    
    val_metrics = []
    
    for i, val_airport in enumerate(airports):

        print(f"fold: {i+1}")

        # Combine the train data (use all airports except the validation one)
        train_airports = [airport for j, airport in enumerate(airports) if j != i]
        
        # Get the training and validation data
        train_data = pd.concat([airport_data_dict[airport] for airport in train_airports])
        train_data = train_data.sort_values(by='DEP_DATE_TIME').dropna()
        val_data = airport_data_dict[val_airport]

        # Preprocess the data (preprocessing should use the same scaler and preprocessor for both training and validation)
        X_train_torch, Y_train_torch, scaler, preprocessor, feature_names = preprocess_no_split(train_data, 'train')
        X_val_torch, Y_val_torch, _, _, feature_names = preprocess_no_split(val_data, 'test', scaler, preprocessor)

        print('data preprocessed')

        # Check if the model function requires `feature_names`
        model_params = inspect.signature(model_name).parameters
        if "feature_names" in model_params:
            model, metrics = model_name(X_train_torch, X_val_torch, Y_train_torch, Y_val_torch, feature_names=feature_names)
        else:
            model, metrics = model_name(X_train_torch, X_val_torch, Y_train_torch, Y_val_torch)

    val_metrics.append(metrics)
    
    return val_metrics

# Get the averages of the metrics by fold
def metric_avgs(val_metrics):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for dict in val_metrics:
        accuracies.append(dict['test']['accuracy'])
        precisions.append(dict['test']['precision'])
        recalls.append(dict['test']['recall'])
        f1s.append(dict['test']['f1'])

    metric_avgs = {}
    metric_avgs['accuracy'] = np.average(accuracies)
    metric_avgs['precision'] = np.average(precisions)
    metric_avgs['recall'] = np.average(recalls)
    metric_avgs['f1'] = np.average(f1s)

    return metric_avgs