# use debian bullseye as the base image to ensure the fixed version of libaom
FROM python:3.12.4-slim-bullseye AS base

# prevents Python from writing pyc files.
# keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# set the working directory in the container
WORKDIR /usr/src/api

# install required system packages and upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libfreetype6-dev \
    libpng-dev \
    && apt-get install -y curl \
    && apt-get install -y procps \
    && python -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# copy requirements.txt from host to container
COPY requirements.txt .

# install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r requirements.txt

# copy the model_data directory and other project files into the container
COPY fastapi_app.py /usr/src/api/fastapi_app.py

# expose the fastapi port
EXPOSE 8000

# start the api server
#CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
#     #--mount=type=bind,source=requirements.txt,target=requirements.txt \