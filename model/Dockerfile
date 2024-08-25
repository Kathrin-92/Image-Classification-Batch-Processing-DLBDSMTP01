# use debian bullseye as the base image to ensure the fixed version of libaom
FROM python:3.12.4-slim-bullseye AS base

# prevents Python from writing pyc files.
# keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# install git and set environment variables
RUN apt-get update && apt-get install -y git
ENV GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git
ENV GIT_PYTHON_REFRESH=quiet

# set the working directory in the container
WORKDIR /usr/src/model

# install required system packages and upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libfreetype6-dev \
    libpng-dev \
    && python -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# copy requirements.txt from host to container
COPY requirements.txt .

# install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install --no-cache-dir -r requirements.txt

# copy the model_data directory and other project files into the container
COPY model_data /usr/src/model/model_data
COPY config_modeltraining.py /usr/src/model/config_modeltraining.py
COPY train_register_cnn_model.py /usr/src/model/train_register_cnn_model.py

# expose the MLflow port
EXPOSE 8080

# start the MLflow server and run the training script
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 8080 --default-artifact-root /usr/src/mlartifacts & python /usr/src/model/train_register_cnn_model.py && tail -f /dev/null"]
