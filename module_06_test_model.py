import pandas as pd
import numpy as np
import ast
from tensorflow import keras

def evaluate_model(model, test_data_path='./datasets/processed_data/pcd_test.csv'):
    """
    Load a trained model, evaluate it on the test dataset, and return accuracies for each label.
    
    Parameters:
    - test_data_path: Path to the test data CSV.
    - model_path: Path to the trained model.
    
    Returns:
    - A list of accuracies (as floats in the range [0, 1]) for each label (0, 1, 2).
    """

    # 1. Load the trained model
    # model = keras.models.load_model(model_path)

    # 2. Load the test dataset
    test_data = pd.read_csv(test_data_path)

    # Assuming the test_data structure is similar to the earlier data
    test_data['features'] = test_data['features'].apply(lambda x: ast.literal_eval(x))
    X_test = pd.DataFrame(test_data['features'].tolist()).values
    y_test = test_data['label'].values

    # 3. Make predictions
    predictions = model.predict(X_test)

    # Assuming a classification task, converting softmax outputs to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # 4. Evaluate the model for overall accuracy
    accuracy = np.mean(predicted_labels == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Evaluate accuracy for each label and store in a list
    accuracies = []
    for label in [0, 1, 2]:
        mask = (y_test == label)
        label_accuracy = np.mean(predicted_labels[mask] == y_test[mask])
        accuracies.append(label_accuracy)
        print(f"Accuracy for label {label}: {label_accuracy:.4f}")

    return accuracies
