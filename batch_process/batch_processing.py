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
import logging

# ----------------------------------------------------------------------------------------------------------------------
# SET UP CONTAINER-SPECIFIC LOGGING
# ----------------------------------------------------------------------------------------------------------------------

# creating log file

log_path = '/usr/src/batch_process'
log_filename = os.path.join(log_path, 'batch_process.log')

# logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


# ----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS FOR LOADING A BATCH OF DATA
# ----------------------------------------------------------------------------------------------------------------------

def load_data(path_to_batch_data, files_already_processed):
    # load and preprocess batch data;
    # check if data was already processed before; read multiple files from temporary storage
    image_data_list = []
    filenames_processed = []
    filenames_list = []

    for filename in os.listdir(path_to_batch_data):
        if filename.endswith(".csv") and filename not in files_already_processed:
            file_path = os.path.join(path_to_batch_data, filename)
            input_data = pd.read_csv(file_path, sep=';')

            # assign the filename to each input_data_row so to match the prediction results to the provided input
            filenames_list.extend([filename] * len(input_data))

            image_data_list.append(input_data)
            filenames_processed.append(filename)
            logger.info(f"New csv-files processed.")
        else:
            if not filename.endswith(".csv"):
                logger.info(f"File '{filename}' is not a .csv file and will be skipped.")
                print(f"File '{filename}' is not a .csv file and will be skipped.")

    if not filenames_processed:
        logger.info("No new data to process...!")
        print("No new data to process...!")
        return pd.DataFrame(), filenames_processed

    df_image_data = pd.concat(image_data_list)

    return df_image_data, filenames_processed, filenames_list


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
    logger.info("Preprocessed new image data (resizing images, normalizing pixel values).")
    return preprocessed_image_data


# ----------------------------------------------------------------------------------------------------------------------
# FUNCTION FOR COMMUNICATION WITH API
# ----------------------------------------------------------------------------------------------------------------------


def get_batch_predictions(preprocessed_data, filenames_list):
    # send batch to api
    endpoint_url = 'http://fastapi:8000/get_batch_prediction'

    try:
        logger.info(f"Sending batch request to {endpoint_url} with {len(filenames_list)} files.")

        response = requests.post(endpoint_url,
                                 json={
                                     "data": preprocessed_data,
                                     "filenames": filenames_list
                                 })

        if response.status_code == 200:
            predictions = response.json()
            logger.info(f"Batch prediction successful. Received {len(predictions)} predictions.")
            return predictions
        else:
            logger.error(f"An error occurred! \nThe status code is: {response.status_code}\nError message: {response.text}")
            print(f"An error occurred! \nThe status code is: {response.status_code}\nError message: {response.text}")
            return []

    except Exception as e:
        logger.exception(f"An unexpected error occurred during the batch prediction: {str(e)}")
        return []


def save_predictions(predictions, output_directory):
    try:
        # save predictions to a file or temporary storage
        logger.info("Starting to save predictions.")
        predictions_df = pd.DataFrame(predictions)
        file_name = f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = os.path.join(output_directory, file_name)
        predictions_df.to_csv(file_path, index=False)
        logger.info(f"Predictions saved successfully to {file_path}.")
        return file_name

    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}", exc_info=True)
        return None


# ----------------------------------------------------------------------------------------------------------------------
# FUNCTION FOR MAIN BATCH PROCESS
# ----------------------------------------------------------------------------------------------------------------------


def batch_process():
    start_time = datetime.now()
    logger.info(f"--->>> Start of Batch Process at {start_time}.<<<---")

    # load and handle batch of image data
    logger.info("Check whether new data is available and load it, if available...")
    print("Check whether new data is available and load it, if available...")
    path_to_data = "/api/upload"
    file_name_processed_files = 'processed_files.json'
    file_path_processed_files = os.path.join(path_to_data, file_name_processed_files)
    files_already_processed = load_processed_files(file_path_processed_files)

    df_image_data, filenames_processed, filenames_list = load_data(path_to_data, files_already_processed)

    if df_image_data.empty:
        logger.info("No new data to process. Exiting batch process.")
        end_time = datetime.now()
        logger.info(f"--->>> End of Batch Process at {end_time}. <<<---")
        return

    else:
        # preprocess image data
        logger.info("Start preprocessing new image data...")
        print("Start preprocessing new image data...")
        preprocessed_image_data = preprocess_data(df_image_data)

        # send preprocessed data to the prediction API
        logger.info("Sending preprocessed data to the prediction API.")
        predictions = get_batch_predictions(preprocessed_image_data, filenames_list)

        if not predictions:
            logger.warning("No predictions were returned by the API.")
            print("No predictions were returned.")
            end_time = datetime.now()
            logger.info(f"--->>> End of Batch Process at {end_time}. <<<---")
        else:
            logger.info("Predictions successful! Proceeding to save results.")
            print("Predictions successful! Save results...")
            path_to_predictions = "/api/prediction-results"
            file_name = save_predictions(predictions, path_to_predictions)
            files_already_processed.extend(filenames_processed)
            save_processed_files(file_path_processed_files, files_already_processed)
            print(f"Results saved in file '{file_name}'.")
            end_time = datetime.now()
            logger.info(f"--->>> End of Batch Process at {end_time}. <<<---")


if __name__ == "__main__":
    batch_process()
