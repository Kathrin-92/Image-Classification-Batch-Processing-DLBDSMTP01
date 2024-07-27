"""
load the CSV file, preprocess the data, and convert it to the required format.
Send a POST request to the FastAPI endpoint with the preprocessed data.
Process the API response and save the predictions to a file or temporary storage.
"""

import requests
import pandas as pd
import numpy as np
from config import FILE_PATH_BATCH


# load and preprocess batch data
input_df = pd.read_csv(FILE_PATH_BATCH, sep=';')
print("Input DataFrame head:\n", input_df.head())

x_input = input_df.iloc[:,:].values / 255.0
x_input = x_input.astype(np.float32)
print("Shape before reshape:", x_input.shape)

x_input = x_input.tolist()  # Convert to list for JSON serialization; Data should be flattened into list[list[float]] for serialization and transmission.
print("reshaping to JSON")

# Send preprocessed data to the prediction API
response = requests.post('http://127.0.0.1:8000/get_batch_prediction', json={"data": x_input})

if response.status_code == 200:
    predictions = response.json()
    print("response status 200")
    # Save predictions to a file or temporary storage
    predictions_df = pd.DataFrame(predictions)
    # add the class names here later on
    predictions_df.to_csv('/Users/kathrinhalbich/PycharmProjects/Image-Classification-Batch-Processing-DLBDSMTP01/batch_prediction_results/results_1.py', index=False)
else:
    print(f"Error: {response.status_code}, {response.text}")