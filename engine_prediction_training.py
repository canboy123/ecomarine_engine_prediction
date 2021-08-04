import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental import preprocessing

#########################################
# TODO
# ----
# 1. Set 'filename_path' as your input file path that you want to train your model
# 2. Set 'col_fuel_name' with the proper header name for your fuel column of your 'filename_path'
# 3. Set 'col_torque_name' with the proper header name for your torque column of your 'filename_path'
# 4. Set 'col_speed_name' with the proper header name for your speed column of your 'filename_path'
# (Optional from step 5 - 6)
# 5. Set 'torque_saved_model_directory' that you want the model to be saved
# 6. Set 'fuel_saved_model_directory' that you want the model to be saved
#########################################

###########################################
#          INITIALIZE VARIABLES           #
###########################################
# Change to the correct path for your dataset, change the header name accordingly
filename_path = "D:/Ho Jiacang/Project 2/project2_dataset.csv"  # Contain a header with 3 column names, which are RF004, MC065N and STW
col_fuel_name = 'RF004'     # The column name of the fuel in your file
col_torque_name = 'MC065N'  # The column name of the torque in your file
col_speed_name = 'STW'      # The column name of the speed in your file
num_of_input = 1
num_of_output = 1
torque_saved_model_directory = 'saved_model/model_torque'   # It will save into the same directory of this script where it is running
fuel_saved_model_directory = 'saved_model/model_fuel'       # It will save into the same directory of this script where it is running

###########################################
#             NEURAL NETWORK              #
###########################################
# A model to predict the torque
def create_torque_model(train_x):
    # Creating normalizer, normalize all the values
    normalizer = preprocessing.Normalization(input_shape=[num_of_input, ], axis=None)
    normalizer.adapt(train_x)

    # Creating the network architecture
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(8),
        layers.Dense(8, activation="relu"),
        layers.Dense(16),
        layers.Dense(units=num_of_output)
    ])

    # model.summary()

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.05),
        loss='mean_squared_error')

    return model

# A model to predict the fuel
def create_fuel_model(train_x):
    # Creating normalizer, normalize all the values
    normalizer = preprocessing.Normalization(input_shape=[num_of_input, ], axis=None)
    normalizer.adapt(train_x)

    # Creating the network architecture
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(8, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(units=num_of_output, activation="relu")
    ])

    # model.summary()

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.03),
        loss='mean_squared_error')

    return model

###########################################
#             OTHER FUNCTION              #
###########################################
# Read a data from csv file with pandas library
def readCsv(filename_path):
    # Dataset is now stored in a Pandas Dataframe
    df = pd.read_csv(filename_path, sep=',', encoding='latin-1')

    # Shuffle the dataset
    df = df.sample(frac=1.0)
    df = df.reset_index()

    return df

# Split the data into training data and test data
def split_into_training_test_data(df):
    data_fuel = df.pop(col_fuel_name)
    data_engine = df.pop(col_torque_name)
    data_speed = df.pop(col_speed_name)

    # Test data is extracted 10% of the full data
    total_rows = len(data_fuel)
    last_row_of_training_data = int(np.ceil(total_rows * 0.9))

    train_speed_x = data_speed[0:last_row_of_training_data]  # Speed input
    train_torque_y = data_engine[0:last_row_of_training_data]  # Engine output
    test_speed_x = data_speed[last_row_of_training_data:total_rows - 1]  # Speed input
    test_torque_y = data_engine[last_row_of_training_data:total_rows - 1]  # Engine output

    train_torque_x = data_engine[0:last_row_of_training_data]  # Engine input
    train_fuel_y = data_fuel[0:last_row_of_training_data]  # Fuel output
    test_torque_x = data_engine[last_row_of_training_data:total_rows - 1]  # Engine input
    test_fuel_y = data_fuel[last_row_of_training_data:total_rows - 1]  # Fuel output

    return train_speed_x, train_torque_y, test_speed_x, test_torque_y, train_torque_x, train_fuel_y, test_torque_x, test_fuel_y


# Function to plot the graph
def plot_graph(pred_x, pred_y, train_x, train_y, test_x, test_y, xlabel, ylabel):
    plt.scatter(train_x, train_y, label='Training Data')
    plt.scatter(test_x, test_y, label='Test Data')
    plt.scatter(pred_x, pred_y, label='Predictions')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

###########################################
#              MAIN FUNCTION              #
###########################################
def main():
    # Reading data from a csv file
    data = readCsv(filename_path)

    # Split the data into training data and test data
    train_speed_x, train_torque_y, test_speed_x, test_torque_y, \
    train_torque_x, train_fuel_y, test_torque_x, test_fuel_y = split_into_training_test_data(data)

    # Create the torque model, train the model, evaluate the model and save the model
    torque_model = create_torque_model(train_speed_x)
    torque_model.fit(train_speed_x, train_torque_y, epochs=1000, batch_size=1000, verbose=1)
    torque_model.evaluate(test_speed_x, test_torque_y, batch_size=128)
    torque_model.save(torque_saved_model_directory)

    # Plot the graph to see the result
    # pred_torque_x = test_speed_x
    # pred_torque_y = torque_model.predict(test_speed_x)
    # plot_graph(pred_torque_x, pred_torque_y, train_speed_x, train_torque_y, test_speed_x, test_torque_y, 'Speed', 'Torque')

    # Create the fuel model, train the model, evaluate the model and save the model
    fuel_model = create_torque_model(train_torque_x)
    fuel_model.fit(train_torque_x, train_fuel_y, epochs=1000, batch_size=1000, verbose=1)
    fuel_model.evaluate(test_torque_x, test_fuel_y, batch_size=128)
    fuel_model.save(fuel_saved_model_directory)

    # Plot the graph to see the result
    # pred_fuel_x = test_torque_x
    # pred_fuel_y = fuel_model.predict(test_torque_x)
    # plot_graph(pred_fuel_x, pred_fuel_y, train_torque_x, train_fuel_y, test_torque_x, test_fuel_y, 'Torque', 'Fuel')

if __name__ == "__main__":
    main()
