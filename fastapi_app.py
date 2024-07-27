from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
import uvicorn

# load the model
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
model_name = "fashion_cnn_model"
model_version = 2
model_uri_path = f"models:/{model_name}/{model_version}"
cnn_model = mlflow.pyfunc.load_model(model_uri=model_uri_path)


app = FastAPI(title="Image Classifier API")


# Define a Pydantic model for input data
class InputData(BaseModel):
    data: list[list[float]]

class PredictionResult(BaseModel):
    input: int
    prediction_class: int


# define the index route
@app.get('/')
async def index():
    return {"Message": "This is a Welcome Message"}

# Define the health check route
@app.get('/health')
async def health_check():
    return {"status": "healthy"}

# Define the batch prediction route
@app.post('/get_batch_prediction', response_model=list[PredictionResult])
async def predict(input_data: InputData):
    try:
        # Convert input data to numpy array
        x_input = np.array(input_data.data, dtype=np.float32)
        print("convert to numpy array")

        # Reshape data to(batch_size, 28, 28, 1)
        x_input = x_input.reshape(x_input.shape[0], 28, 28, 1)
        print('reshape data to model requirements')

        # Make predictions
        predictions = cnn_model.predict(x_input)
        print("try to make predictions")
        predicted_classes = predictions.argmax(axis=1)
        print("get predictions")

        # Prepare the results
        results = [{"input": i + 1, "prediction_class": int(pred)} for i, pred in enumerate(predicted_classes)]

        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input data: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, log_level="debug", reload=True)

# http://127.0.0.1:8000/docs/