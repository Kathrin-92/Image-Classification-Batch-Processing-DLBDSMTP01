"""
Script that provides the functionalities for the batch process that sends image data to an API to obtain probabilities
for the assignment to an image class.
The API serves as an internal service that the batch processing job interacts with to obtain predictions from the
cnn model registered with mlflow.

The script provides functions that load a batch of CSV files (aka image data), preprocess the data, and convert it to
the required format for the API. It also sends a POST request to the FastAPI endpoint with the preprocessed data.
The response is received and the predictions saved to a file into a local directory. The prediction results are
stored in a file with the name prediction_results_timestamp.csv.
"""

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# standard library imports
import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# local imports
from config import BASE_PATH_BATCH, PROCESSED_FILES_PATH, PREDICTIONS_PATH


# ----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS FOR LOADING A BATCH OF DATA
# ----------------------------------------------------------------------------------------------------------------------


def load_data(base_path_to_batch_data, files_already_processed):
    # load and preprocess batch data;
    # check if data was already processed before; read multiple files from temporary storage
    image_data_list = []
    filenames_processed = []

    for filename in os.listdir(base_path_to_batch_data):
        if filename.endswith(".csv") and filename not in files_already_processed:
            file_patch_batch = os.path.join(base_path_to_batch_data, filename)
            input_data = pd.read_csv(file_patch_batch, sep=';')
            # give each input_data_row a unique name/id so that later i can better match the results to the provided input
            image_data_list.append(input_data)
            filenames_processed.append(filename)
        else:
            if not filename.endswith(".csv"):
                print(f"File '{filename}' is not a .csv file and will be skipped.")

    if not filenames_processed:
        print("No new data to process...!")
        return pd.DataFrame(), filenames_processed

    df_image_data = pd.concat(image_data_list)
    return df_image_data, filenames_processed


def load_processed_files(filepath):
    # check if there is already a list of already processed files; load or create it
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        with open(filepath, 'w') as f:
            json.dump([], f)
        return []


def save_processed_files(filepath, files_list):
    # save new processed image data to list
    with open(filepath, 'w') as f:
        json.dump(files_list, f)


# ----------------------------------------------------------------------------------------------------------------------
# FUNCTION FOR PREPROCESSING LOADED BATCH DATA
# ----------------------------------------------------------------------------------------------------------------------


def preprocess_data(df_image_data):
    # preprocess image data (resizing images, normalizing pixel values)
    df_image_data = df_image_data.values / 255.0
    df_image_data = df_image_data.astype(np.float32)
    # convert to list for JSON serialization; data needs to be flattened into list[list[float]]
    preprocessed_image_data = df_image_data.tolist()
    return preprocessed_image_data


# ----------------------------------------------------------------------------------------------------------------------
# FUNCTION FOR COMMUNICATION WITH API
# ----------------------------------------------------------------------------------------------------------------------


def get_batch_predictions(preprocessed_data):
    # send batch to api
    endpoint_url = 'http://127.0.0.1:8000/get_batch_prediction'
    response = requests.post(endpoint_url, json={"data": preprocessed_data})

    if response.status_code == 200:
        predictions = response.json()
        return predictions
    else:
        print(f"An error occurred! \nThe status code is: {response.status_code}\nError message: {response.text}")
        return []


def save_predictions(predictions, output_directory):
    # save predictions to a file or temporary storage
    predictions_df = pd.DataFrame(predictions)
    file_name = f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    file_path = os.path.join(output_directory, file_name)
    predictions_df.to_csv(file_path, index=False)
    return file_name


# ----------------------------------------------------------------------------------------------------------------------
# FUNCTION FOR MAIN BATCH PROCESS
# ----------------------------------------------------------------------------------------------------------------------


def batch_process():
    # load and handle batch of image data
    print("Check whether new data is available and load it, if available...")
    file_name_processed_files = 'processed_files.json'
    file_path_processed_files = os.path.join(PROCESSED_FILES_PATH, file_name_processed_files)
    files_already_processed = load_processed_files(file_path_processed_files)

    df_image_data, filenames_processed = load_data(BASE_PATH_BATCH, files_already_processed)

    if df_image_data.empty:
        return
    else:
        # preprocess image data
        print("Start preprocessing new image data...")
        preprocessed_image_data = preprocess_data(df_image_data)

        # send preprocessed data to the prediction API
        predictions = get_batch_predictions(preprocessed_image_data)

        if not predictions:
            print("No predictions were returned.")
        else:
            print("Predictions successful! Save results...")
            file_name = save_predictions(predictions, PREDICTIONS_PATH)
            files_already_processed.extend(filenames_processed)
            save_processed_files(file_path_processed_files, files_already_processed)
            print(f"Results saved in file '{file_name}'.")


if __name__ == "__main__":
    batch_process()
