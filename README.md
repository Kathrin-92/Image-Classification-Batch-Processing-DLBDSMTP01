# Project: CNN Model Deployment with FastAPI and Batch Processing

## Table of Contents
1. [General Info](#General-Info)
2. [Application Setup and Execution Guide](#Application-Setup-and-Execution-Guide)


## General Info

In this project, the processing and deployment of a machine learning model is simulated. There are several essential elements in the setup: 
Using the Fashion MNIST dataset, a rudimentary machine learning model is created to automatically categorize return items into categories 
based on pictures of the products. This dataset, provided in ready-made CSV files, is available for download [here](https://www.kaggle.com/datasets/zalando-research/fashionmnist/data). After registering with MLFlow, 
the model is made available as a service that can be started overnight in batch operations. To enable connection between the batch processing 
system and the MLFlow server, FastAPI is used. Dummy image files (CSV files) are handled by the batch process, which is started nightly by 
a cron job and runs from a specified directory. 

All three components—MLFlow/model, FastAPI, and the batch process—are packaged as Docker containers to streamline the deployment and ensure a consistent production environment.

The application includes three main components: 
- **MLFlow Server** (`mlflow_container`): Runs the CNN model and provides access to the MLFlow server on port 8080.
- **FastAPI** (`fastapi_container`): Provides API functionalities. The Fastapi server runs on port 8000.
- **Batch Process** (`batch_process_container`): Automates the daily processing of CSV files at midnight.



## Application Setup and Execution Guide
### Prerequisites
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop) or install Docker directly on your machine
2. Clone this Git repository
3. Navigate into the cloned repository directory:
    ```
    cd <repository-directory>
    ```
4. Ensure that the `docker-compose.yaml` file is present in the root of the project
5. (! important !) Go to Kaggle and download the data for the model training [here](https://www.kaggle.com/datasets/zalando-research/fashionmnist/data): unpack the .zip file and place the two files "fashion-mnist_train.csv" and "fashion-mnist_test.csv" into the directory model/model_data



### Step-by-Step Guide
#### Step 1: Build and Run the MLFlow Server / CNN Model

```
docker compose -f compose.yaml up -d mlflow
```

- The container is named `mlflow_container`.
- Once started, it takes about **5 minutes** for the model to train.
- Access the MLFlow server at **http://localhost:8080** to check your trained and registered model.

#### Step 2. Build and Run the FastAPI Docker Container

```
docker compose -f compose.yaml up -d fastapi
```

- The container is named `fastapi_container`.
- Access the Fastapi server at **http://localhost:8000/docs** to check the Fastapi documentation.

#### Step 3. Build and Run the Batch Process

```
docker compose -f compose.yaml up -d batch_process
```

- The container is named `batch_process_container`.
- After the first run of this container, two new folders will be created in the directory where you cloned the repository:
   - `upload`
   - `prediction-results`

   Place the `.csv` files you want to process into the `upload` folder.
- The batch process will automatically process all the files in the `upload` folder at **00:00 (midnight)** every day.
- Test the process using the example data in the `dummy_upload_data.zip` file.
