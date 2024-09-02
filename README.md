# Project: CNN Model Deployment with FastAPI and Batch Processing

## Table of Contents
1. [General Info](#General-Info)
2. [Application Setup and Execution Guide](#Application-Setup-and-Execution-Guide)
3. [Usage and Main Functionalities](#Usage-and-Main-Functionalities)



## General Info
Lorem Ipsum

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



## Usage and Main Functionalities
Lorem Ipsum