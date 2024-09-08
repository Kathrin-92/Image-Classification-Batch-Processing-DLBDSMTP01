"""
Script that loads the Fashion MNIST dataset, uses Keras to train a basic Convolutional Neural Network (CNN),
and saves the trained model.
The Fashion MNIST dataset, which consists of grayscale photos of clothing items, is loaded and preprocessed after the
required libraries have been imported. Using Keras, the script builds a simple CNN model, then trains it on the dataset
to categorize the photos into several fashion categories.

The python script is based on the following tutorials and the course script for Neural Nets and Deep Learning:
https://www.kaggle.com/code/vishwasgpai/guide-for-creating-cnn-model-using-csv-file
https://www.kaggle.com/code/pavansanagapati/a-simple-cnn-model-beginner-guide/notebook

After evaluating the model's performance,the trained model is stored to the MLflow model registry for later reuse.
The script imports the necessary libraries, including MLflow, loads the trained model and logs the model to MLflow,
specifying relevant metadata such as the model name and version.
With this, it registers the model in the MLflow model registry, making it available for deployment.

"""

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------

# standard library imports
import pandas as pd
import numpy as np

# third-party imports
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from mlflow.models import infer_signature
import mlflow.keras

# local imports
from config_modeltraining import FILE_PATH_TRAIN, FILE_PATH_TEST


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
test_df = pd.read_csv(FILE_PATH_TEST, sep=',')
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
sample_data_visualization = plt.gcf()

# reshape training, validation and test data, so that it can be processed by keras model;
# to fit expected input shape of the convolutional layers in the CNN model, data is reshaped into a 3D format/matrix
# convolutional layers in Keras expect input data in the shape of (height, width, channels)
# for a grayscale image, channels is 1
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_validate = x_validate.reshape(x_validate.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# ----------------------------------------------------------------------------------------------------------------------
# SETUP AND TRAIN MODEL
# ----------------------------------------------------------------------------------------------------------------------

# define CNN model architecture
cnn_model = Sequential([
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(28, 28, 1)),  # batch size is handled implicitly by keras and does not need to be specified
    MaxPooling2D(pool_size=(2, 2)),  # first max pooling layer
    Dropout(rate=0.25),  # dropout layer with 25% dropout rate; added after each pooling layer to reduce overfitting
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(rate=0.25),  # dropout layer with 25% dropout rate; added after each pooling layer to reduce overfitting
    Flatten(),  # flatten output
    Dense(
        units=128,
        activation='relu'),  # fully connected layer
    Dropout(rate=0.5),  # dropout layer with 50% dropout rate;
                        # added before dense layer to prevent overfitting on fully connected layers
    Dense(
        units=10,
        activation='softmax')  # output layer
])

# compile the model
cnn_model.compile(
    optimizer=Adam(learning_rate=0.001),  # how model is updated based on the data it sees and its loss function;
                                          # adam adapts learning rates and stabilizes training; default learning rate
    loss='sparse_categorical_crossentropy',  # more memory-efficient training with integer labels
    metrics=['accuracy']
)

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


# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE MODEL
# ----------------------------------------------------------------------------------------------------------------------
cnn_model.summary()

test_loss, test_acc = cnn_model.evaluate(x_test, y_test)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')


# ----------------------------------------------------------------------------------------------------------------------
# SIGNATURE & LOG MODEL WITH MLFLOW
# ----------------------------------------------------------------------------------------------------------------------

# set the tracking URI
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# create sample of input and output data so that signature can be automatically inferred
sample_input = x_test[:1]
sample_output = cnn_model.predict(sample_input)
signature_mlflow = infer_signature(sample_input, sample_output)

# log model with MLflow / save model for later retrieval
with mlflow.start_run() as run:
    mlflow.set_tag("Training Info", "Basic cnn model for fashion mnist dataset.")
    model_info = mlflow.keras.log_model(
        model=cnn_model,
        artifact_path="fashion_cnn_model_artifacts",
        signature=signature_mlflow,
        registered_model_name="fashion_cnn_model"
    )
    mlflow.log_metric("Test accuracy", test_acc)  # log training statistics
    mlflow.log_metric("Test loss", test_loss)  # log training statistics
    mlflow.log_figure(sample_data_visualization, "sample_data_visualization.png")

artifact_uri = run.info.artifact_uri
run_id = run.info.run_id
model_uri = f"runs:/{run_id}/cnn_model"
print(artifact_uri)


# ----------------------------------------------------------------------------------------------------------------------
# MAKE PREDICTIONS USING TESTING DATA
# ----------------------------------------------------------------------------------------------------------------------

# load the model back for predictions as a generic Python Function model
cnn_model = mlflow.pyfunc.load_model(model_info.model_uri)

# get the predictions for the test data
# each element in array represents predicted probability for a class; first value = probability of class 0
predictions = cnn_model.predict(x_test)
# convert predictions from probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)

# print some example predictions for one case to get a feel for the results
print("Prediction is -> {}".format(predictions[12]))
print("Actual value is -> {}".format(y_test[12]))
print("The highest value for label is {}".format(predicted_labels[12]))

# Start a new run to log additional artifacts
with mlflow.start_run(run_id=run_id, nested=True) as run:
    # visualize some test images along with their predicted and actual labels
    plt.figure(figsize=(10, 10))
    for i in range(25):  # number of images to display
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        true_label = class_names[int(y_test[i])]
        predicted_label = class_names[predicted_labels[i]]
        plt.title(f'Pred: {predicted_label}\nTrue: {true_label}')
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.savefig('test_data_visualization.png')
    mlflow.log_artifact('test_data_visualization.png')

