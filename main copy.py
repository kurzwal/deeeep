from module_06_test_model import evaluate_model
from module_05_train_model import train_existing_model
from module_04_build_model import build_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import ast
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import time



if __name__ == "__main__":
    # =====================
    # Constants
    num_iterations = 5000
    max_neurons = 513
    min_neurons = 32
    num_hidden_layers = 3
    # =====================
    save_path = "./models/"
    save_extention = ".keras"

    model_name = "test_model_0"  # Initial model name

    data = pd.read_csv('./datasets/processed_data/cleaned_pcd_all.csv')
    data['features'] = data['features'].apply(lambda x: ast.literal_eval(x))
    X = pd.DataFrame(data['features'].tolist()).values
    y = data['label'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # prev_time = time.time()

    for i in range(num_iterations):

        # Randomly generating neuron_layer
        neuron_layer = list(np.random.randint(
            min_neurons, max_neurons, size=num_hidden_layers))
        neuron_layer.sort(reverse=True)  # Sort in descending order
        print(f"Iteration {i+1}: neuron_layer: {neuron_layer}")

        # Building and saving the model
        model = build_model((1440, ), neuron_layer)
        # print("save model")
        # model.save(save_path + model_name + save_extention)
        # print("model saved")

        # Train the model
        trained_model = train_existing_model(
            model, X_train, X_val, y_train, y_val)
        # Evaluate the model
        evaluation = evaluate_model(trained_model)
        # current_time = time.time()
        # time_difference = current_time - prev_time
        print("")
        # If evaluation results are all above 0.3, change the model name for the next iteration and update the model.txt
        if all([ev >= 0.3 for ev in evaluation]):
            with open('./models/model.txt', 'a') as file:
                file.write(f"{model_name}.keras\n")
                file.write(f"   case {i}\n")
                file.write(f"   neuron_layer: {neuron_layer}\n")
                file.write(f"   Test Accuracy: {evaluation[0]:.4f}\n")
                # Assumes evaluate_model returns test accuracy as the first element
                for idx, ev in enumerate(evaluation[1:], start=1):
                    file.write(f"   Accuracy for label {idx}: {ev:.4f}\n")
                file.write("\n")
            trained_model.save(save_path + model_name + save_extention)
            model_name = f"test_model_{i + 1}"
