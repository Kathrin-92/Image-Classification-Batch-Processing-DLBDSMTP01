"""
The logic for the API endpoint that can receive picture data, parse it, and deliver predictions is contained in this script.

It manages the subsequent actions:
1. A NumPy array is created from the input data.
2. To conform to the CNN model's necessary input format, the array is reshaped.
3. Utilizing the CNN model, predictions are created.
4. For every input, the highest class and associated probability are retrieved.
5. After mapping the predictions to the appropriate class names, they are given back.
"""

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# standard library imports
import numpy as np

# third-party imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
# import uvicorn # for local run


# ----------------------------------------------------------------------------------------------------------------------
# LOADING THE TRAINED MODEL FROM MLFLOW
# ----------------------------------------------------------------------------------------------------------------------

# load the model
mlflow.set_tracking_uri(uri="http://mlflow:8080")
model_name = "fashion_cnn_model"
model_version = 1

try:
    model_uri = f"models:/{model_name}/{model_version}"
    cnn_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    print(f"Model loaded from {model_uri}")
except MlflowException as e:
    print(f"Failed to load model: {str(e)}")

# ----------------------------------------------------------------------------------------------------------------------
# START API AND DEFINE INPUT AND OUTPUT
# ----------------------------------------------------------------------------------------------------------------------

# for local run
#def run_server():
#    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="debug", reload=True)
#    server = uvicorn.Server(config)
#    server.run()

app = FastAPI(title="Image Classifier API")


# define model for input data
class InputData(BaseModel):
    data: list[list[float]]
    filenames: list[str]

# define model for output data
class PredictionResult(BaseModel):
    input: int
    filename: str
    most_likely_class_name: str
    most_likely_class_number: int
    corresponding_probability: float


# ----------------------------------------------------------------------------------------------------------------------
# DEFINE API ENDPOINTS
# ----------------------------------------------------------------------------------------------------------------------

# define the index route
@app.get('/')
async def index():
    return {
        "Message": "Welcome to the Image Classification API! This API connects a Convolutional Neural Network (CNN) model to a batch process that returns predictions for the most likely class based on image input.",
        "Instructions": "To explore the API's documentation and available endpoints, please visit: http://localhost:8000/docs"
    }


# define the health check route
@app.get('/health')
async def health_check():
    return {"status": "healthy"}


# define the batch prediction route
@app.post('/get_batch_prediction', response_model=list[PredictionResult])
async def predict(input_data: InputData):
    try:
        # convert input data to numpy array
        x_input = np.array(input_data.data, dtype=np.float32)

        # reshape data to model requirements (batch_size, 28, 28, 1)
        x_input = x_input.reshape(x_input.shape[0], 28, 28, 1)

        # make predictions
        predictions = cnn_model.predict(x_input)

        # get the highest class and probability
        highest_class_number = predictions.argmax(axis=1)
        corresponding_probability = predictions.max(axis=1)
        print(highest_class_number)

        # prepare the results
        class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        results = [{
            "input": i + 1,
            "filename": input_data.filenames[i],
            "most_likely_class_name": class_names[highest_class_number[i]],
            "most_likely_class_number": int(highest_class_number[i]),
            "corresponding_probability": float(corresponding_probability[i])
        } for i in range(len(highest_class_number))]

        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input data: {str(e)}")

# for local run
#if __name__ == "__main__":
#   run_server()
