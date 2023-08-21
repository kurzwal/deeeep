from keras.models import Sequential
from keras.layers import Dense, Dropout

def build_model(input_shape, hidden_layers):
    """
    Build and compile a simple classification model.

    Parameters:
    - input_shape: Shape of the input data (e.g., (1440, ))
    - hidden_layers: List containing the number of neurons in each hidden layer.

    Returns:
    - model: Compiled Keras model
    """

    model = Sequential()

    # Input layer
    model.add(Dense(1440, activation='tanh', input_shape=input_shape))

    # Hidden layers
    for neurons in hidden_layers[0:]:
        model.add(Dense(neurons, activation='relu'))
        model.add(Dropout(0.2))

    # Output layer for classification
    # Assuming 3 classes: -1, 0, 1, hence using 'softmax' and 3 neurons
    model.add(Dense(3, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
