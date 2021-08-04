import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn import preprocessing
import random
import csv
from pathlib import Path

#########################################
# TODO
# ----
# 1. Set 'input_data_file' as your input file path that you want to predict the speed of the vessel
# 2. Set 'col_name_speed' with the proper header name for your speed column of your 'input_data_file'
# 3. Set 'torque_saved_model_directory' that you have saved from the 'engine_prediction_training.py' script that you have run
# 4. Set 'fuel_saved_model_directory' that you have saved from the 'engine_prediction_training.py' script that you have run
# (Optional from step 5 - 7)
# 5. Set 'header_request_from_csv_file' into a proper header name according to your 'input_data_file' header
# 6. Set 'header_for_output_csv_file' that you have saved from the 'engine_prediction_training.py' script that you have run
# 7. Set 'output_csv_file' as the output name of the csv file
#########################################


###########################################
#          INITIALIZE VARIABLES           #
###########################################
input_data_file = 'D:/Ho Jiacang/Project 2/data_filtered_ships.csv' # Input data file path
col_name_speed = "sog"  # The column name of the speed in your file
torque_saved_model_directory = 'D:/Ho Jiacang/Project 2/saved_model/model_torque'   # The path where your torque model has saved
fuel_saved_model_directory = 'D:/Ho Jiacang/Project 2/saved_model/model_fuel'       # The path where your fuel model has saved
header_request_from_csv_file = ["gid", "mmsi", "imo", "sog"]    # The column name that you need from your csv file
header_for_output_csv_file = ["gid", "mmsi", "imo", "sog", "torque", "fuel"]    # The header name for the output of your csv file
destination_path = 'csv/'   # The output path for your output csv file
output_csv_file = "output_filename.csv"     # The output of the csv file name

###########################################
#             NEURAL NETWORK              #
###########################################
# Function to predict torque value by using speed value
def predicting_torque(sog_data):
    model = tf.keras.models.load_model(torque_saved_model_directory)
    preds_torque = model.predict(sog_data)

    return preds_torque

# Function to predict fuel consumption by using torque value
def predicting_fuel(preds_torque):
    model = tf.keras.models.load_model(fuel_saved_model_directory)
    preds_fuel = model.predict(preds_torque)

    return preds_fuel

###########################################
#             OTHER FUNCTION              #
###########################################
# Read a data from csv file with pandas library
def readCsv(filename_path):
    # Dataset is now stored in a Pandas Dataframe
    df = pd.read_csv(filename_path, sep=',', encoding='latin-1')

    # Shuffle the dataset
    # df = df.sample(frac=1.0)
    # df = df.reset_index()

    return df

# Used to write a csv file
def writeCSV(output_filename, header, outputs, write_type="w"):
    print("Writing to a csv file...")
    # Create the folder if it is not existed
    Path(destination_path).mkdir(parents=True, exist_ok=True)
    with open(destination_path+output_filename, write_type, encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        for tmp_output in outputs:
            writer.writerow(tmp_output)
    print("Finish of writing.")

# Used to plot a graph
def plot_graph(sog, torque, fuel):
    sog = sog.to_numpy()

    # Normalize the data
    sog = preprocessing.normalize(np.reshape(sog, (1, -1)))
    torque = preprocessing.normalize(np.reshape(torque, (1, -1)))
    fuel = preprocessing.normalize(np.reshape(fuel, (1, -1)))

    sns.distplot(sog, label="sog")
    sns.distplot(torque, label="torque")
    sns.distplot(fuel, label="fuel")

    plt.xlabel('Scaling')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    plt.clf()

###########################################
#              MAIN FUNCTION              #
###########################################
# The main code is running here
def main():
    # Read data from a csv file
    df = readCsv(input_data_file)

    # Only speed data columns are extracted
    sog_data = df[col_name_speed]

    # Get the prediction for torque and fuel values
    preds_torque = predicting_torque(sog_data)
    preds_fuel = predicting_fuel(preds_torque)

    preds_torque = np.reshape(preds_torque, (-1))
    preds_fuel = np.reshape(preds_fuel, (-1))

    # Plot the graph
    # plot_graph(sog_data, preds_torque, preds_fuel)

    # WRITE THE OUTPUT INTO CSV FILE
    # UNCOMMENT THE CODE BELOW IF YOU NEED TO WRITE THEM INTO CSV FILE

    # included_df = df[header_request_from_csv_file].copy() # The data that you want to be used later
    # included_df["torque"] = preds_torque  # Add two columns into existing dataframes
    # included_df["fuel"] = preds_fuel
    #
    # outputs = []
    # for i in range(len(included_df.values)):
    #     outputs.append(included_df.values[i])
    #
    # writeCSV(output_csv_file, header_for_output_csv_file, outputs)    # Write the output into a csv file

if __name__ == "__main__":
    main()
