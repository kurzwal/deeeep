from sklearn.model_selection import train_test_split


def train_existing_model(model, X_train, X_val, y_train, y_val, batch_size=32, epochs=10):
    """
    Load an existing model, train it on provided data, and save the trained model.

    Parameters:
    - data_name: Name of the csv file without extension.
    - model_path: Path to load the existing model.
    - save_path: Path to save the trained model.
    - batch_size: Training batch size.
    - epochs: Number of training epochs.
    """

    # 학습
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              batch_size=batch_size, epochs=epochs, verbose=0)

    print("Trained model saved successfully!")
    return model
