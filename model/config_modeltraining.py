# Standard Library Imports
import os

# Defining FILE_PATH variable to test and train data for model training
BASE_PATH = '/usr/src/model/model_data'
FILE_NAME_TRAIN = 'fashion-mnist_train.csv'
FILE_NAME_TEST = 'fashion-mnist_test.csv'
FILE_PATH_TRAIN = os.path.join(BASE_PATH, FILE_NAME_TRAIN)
FILE_PATH_TEST = os.path.join(BASE_PATH, FILE_NAME_TEST)
