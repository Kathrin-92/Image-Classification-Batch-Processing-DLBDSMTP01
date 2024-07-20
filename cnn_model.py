# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# standard library imports
import pandas as pd
import numpy as np

# third-party imports
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# local imports
from config import FILE_PATH_TRAIN, FILE_PATH_TEST

# ----------------------------------------------------------------------------------------------------------------------
# LOADING DATA
# ----------------------------------------------------------------------------------------------------------------------

# download dataset here: https://www.kaggle.com/datasets/zalando-research/fashionmnist/data
# read training data file
train_df = pd.read_csv(FILE_PATH_TRAIN, sep=',')
# checking data structure
train_df.info()  # -->  60000 entries; 785 entries, label to pixel784; dataype = int
train_df.head()  # --> first column is the label data, the remaining columns are pixel data

# read testing data file
test_df = pd.read_csv(FILE_PATH_TEST, sep=';')
# checking data structure
test_df.info()  # -->   9901 entries; 785 entries, label to pixel784; dataype = int
test_df.head()  # --> first column is the label data, the remaining columns are pixel data


# ----------------------------------------------------------------------------------------------------------------------
# PREPROCESSING DATA
# ----------------------------------------------------------------------------------------------------------------------

# split train and test data into x and y arrays where x represents image data and y represents labels
train_data = np.array(train_df, dtype='float32')
test_data = np.array(test_df, dtype='float32')

# normalize data from 0 to 1 by dividing by 255, because max value in the matrix is 255
x_train = train_data[:, 1:]/255  # actual data; includes every row and pixel-columns without label and header info
y_train = train_data[:, 0]  # label info; all rows and first column
x_test = test_data[:, 1:]/255
y_test = test_data[:, 0]

# split training data into validation and training set
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# visualize some sample data for 2D plotting
class_names = ['T_shirt/top (0)', 'Trouser (1)', 'Pullover (2)', 'Dress (3)', 'Coat (4)',
               'Sandal (5)', 'Shirt (6)', 'Sneaker (7)', 'Bag (8)', 'Ankle boot (9)']
plt.figure(figsize=(10, 10))
for i in range(25):  # create a 5x5 grid (25 images)
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i].reshape((28, 28)), cmap='gray')
    label_index = int(y_train[i])
    plt.title(class_names[label_index])
# plt.show()

# reshape training, validation and test data, so that it can be processed by keras model;
# to fit expected input shape of the convolutional layers in the CNN model, data is reshaped into a 3D format/matrix
# convolutional layers in Keras expect input data in the shape of (height, width, channels)
# for a grayscale image, channels is 1
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_validate = x_validate.reshape(x_validate.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print("done reshaping")

# ----------------------------------------------------------------------------------------------------------------------
# SETUP AND TRAIN MODEL
# ----------------------------------------------------------------------------------------------------------------------

# define CNN model architecture
cnn_model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),  # first convolutional layer
    MaxPooling2D(pool_size=(2, 2)),  # first max pooling layer
    Dropout(rate=0.25),  # dropout layer with 25% dropout rate; added after each pooling layer to reduce overfitting
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(rate=0.25),  # dropout layer with 25% dropout rate; added after each pooling layer to reduce overfitting
    Flatten(),  # flatten output
    Dense(units=128, activation='relu'),  # fully connected layer
    Dropout(rate=0.5),  # dropout layer with 50% dropout rate;
                        # added before dense layer to prevent overfitting on fully connected layers
    Dense(units=10, activation='softmax')  # output layer
])
print("done model architecture setup")

# compile the model
cnn_model.compile(
    optimizer=Adam(learning_rate=0.001),  # how model is updated based on the data it sees and its loss function;
                                          # adam adapts learning rates and stabilizes training; default learning rate
    loss='sparse_categorical_crossentropy',  # more memory-efficient training with integer labels
    metrics=['accuracy']
)
print("done compiling")

# configure early stopping; helps to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss',  # monitor validation loss
                               patience=3,  # epochs to wait for improvement
                               restore_best_weights=True)  # restore best weights after stopping


# train the model
cnn_model.fit(
    x_train,
    y_train,
    batch_size=64,  # number of samples processed before model’s parameters are updated 128, 256
                    # batch size affects training speed, stability, and generalization
    epochs=10,  # number of times the entire dataset is passed through the network during training
    validation_data=(x_validate, y_validate),
    callbacks=[early_stopping]
)
print("done training")

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE MODEL
# ----------------------------------------------------------------------------------------------------------------------
cnn_model.summary()

test_loss, test_acc = cnn_model.evaluate(x_test, y_test)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')
