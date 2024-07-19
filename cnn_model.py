# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# Standard Library Imports
from config import FILE_PATH_TRAIN, FILE_PATH_TEST
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# LOADING DATA
# ----------------------------------------------------------------------------------------------------------------------

# Download dataset here: https://www.kaggle.com/datasets/zalando-research/fashionmnist/data
# Read training data file
train_df = pd.read_csv(FILE_PATH_TRAIN, sep=',')
# Checking data structure
train_df.info()  # -->  60000 entries; 785 entries, label to pixel784; dataype = int
train_df.head()  # --> first column is the label data, the remaining columns are pixel data

# Read testing data file
test_df = pd.read_csv(FILE_PATH_TEST, sep=';', dtype={11: str})
# Checking data structure
test_df.info()  # -->  60000 entries; 785 entries, label to pixel784; dataype = int
test_df.head()  # --> first column is the label data, the remaining columns are pixel data


# ----------------------------------------------------------------------------------------------------------------------
# PREPROCESSING DATA
# ----------------------------------------------------------------------------------------------------------------------

# split train and test data into x and y arrays where x represents image data and y represents labels
train_data = np.array(train_df, dtype='float32')
test_data = np.array(test_df, dtype='float32')

# rescale data from 0 to 1 by dividing by 255
x_train = train_data[:, 1:]/255  # actual data; includes every row and pixel-columns without label and header info
y_train = train_data[:, 0]  # label info; all rows and first column
x_test = test_data[:, 1:]/255
y_test = test_data[:, 0]

# split training data into validation and training set
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# visualize some sample data
class_names = ['T_shirt/top (0)', 'Trouser (1)', 'Pullover (2)', 'Dress (3)', 'Coat (4)',
               'Sandal (5)', 'Shirt (6)', 'Sneaker (7)', 'Bag (8)', 'Ankle boot (9)']
plt.figure(figsize=(10, 10))
for i in range(25):  # create a 5x5 grid (25 images)
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i].reshape((28,28)), cmap='gray')
    label_index = int(y_train[i])
    plt.title(class_names[label_index])
plt.show()
