"""Updates the measurment file """
from __future__ import print_function
import cv2
import numpy as np
import glob
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from Misc.misc import get_image, get_data_paths, get_image_name
from Spatiotemporal.data_configuration import Config
from Misc.preprocessing import get_one_hot_encoded
from Spatial.data_configuration import Config

MEASURMENT_PATH = "/Measurments/recording.csv"
conf = Config()

def extend_steering_commands(dataframe):
    """
    For each intersection, this function will change the 10 previos directions.
    Similar to a person getting a direction before he reaches the intersection
    """
    for index, row in dataframe.iterrows():
        if row.Direction != "RoadOption.LANEFOLLOW" and row.Direction != "RoadOption.VOID":
            rows_to_change = 6
            if index > rows_to_change:
                for i in range(index-6, index):
                    if dataframe.loc[i, "Direction"] != "RoadOption.LANEFOLLOW" and \
                        dataframe.loc[i, "Direction"] != row.Direction:
                        rows_to_change = (index - 1) - i
                dataframe.at[range(index-rows_to_change, index), "Direction"] = row.Direction
    return dataframe

def one_hot_encode_fields(df):
    ohe_directions = get_one_hot_encoded(conf.direction_categories, df.Direction)
    ohe_tl_state = get_one_hot_encoded(conf.tl_categories, df.TL_state)
    ohe_speed_limits = get_one_hot_encoded(conf.sl_categories, df.speed_limit)
    df.loc[:, "ohe_speed_limit"] = None
    df['ohe_speed_limit'] = df['ohe_speed_limit'].astype(object)

    for index, _ in df.iterrows():
        df.at[index, "Direction"] = ohe_directions[index]
        df.at[index, "TL_state"] = ohe_tl_state[index]
        try:
            df.at[index, "ohe_speed_limit"] = ohe_speed_limits[index]
        except ValueError as v:
            print("VALUEERROR" + str(v))
            print(ohe_speed_limits[index])
            exit()
def remove_unused_collums(df):
    input_measures = [
        key for key in conf.available_columns if conf.input_data[key]
        ]
    output_measures = [
        key for key in conf.available_columns if conf.output_data[key]
        ]
    measures = input_measures + output_measures + ["frame"]

    return df[measures]


def pad_frame(df):
    df.frame = df.frame.astype(str)
    for i, frame in enumerate(df.frame):
        df.at[i, "frame"] = get_image_name(frame).split(".")[0]

def round_off_values(df, measure, decimals):
    df[measure] = np.round(df[measure], decimals)

def update_measurements(path):
    DATA_PATHS = get_data_paths(path)
    start_index = 0
    for path in DATA_PATHS[start_index:]:
        print("\r" + path)
        df = pd.read_csv(path + MEASURMENT_PATH)
        if df.iloc[0].Steer == 0 and df.iloc[0].Brake == 0 and df.iloc[0].Throttle == 0:
            df = df.drop(index=0) #remove two first since the car is in the air 
            df = df.reset_index()    
        df = extend_steering_commands(df)
        #print(path)
        one_hot_encode_fields(df)
        pad_frame(df)
        round_off_values(df, "Steer", 3)
        round_off_values(df, "Throttle", 3)
        round_off_values(df, "Speed", 3)
        df = remove_unused_collums(df)
        df.to_csv(path + "/Measurments/modified_recording.csv", index=False)


def update_images(dir_path="../../Training_data_temp", cur_folder=None, file_name=None):
    CONF = Config()
    MEASURMENT_PATH = "/Measurments/recording.csv"
    DATA_PATHS = []
    if cur_folder == None:
        DATA_PATHS = get_data_paths(dir_path)
    else:
        DATA_PATHS.append(dir_path + "/" + str(cur_folder))

    for path in DATA_PATHS:
        print(path)
        try:
            os.mkdir(path + "/Updated_images/")  
        except OSError as ose:
            print("Folder allready exists")
        df = pd.read_csv(path + MEASURMENT_PATH)
        for j, row in df.iterrows():
            cur_frame = row["frame"]
            img = get_image(path + "/Images/", cur_frame)
            img_size = CONF.input_size_data["Image"]
            img = img[CONF.top_crop:, :, :]
            img = cv2.resize(img,(img_size[1], img_size[0]))
            cv2.imwrite(path + "/Updated_images/" + get_image_name(cur_frame), img)


root_folder = "Validation_data"
paths = os.listdir("../../" + root_folder)
for p in paths:
    if "no_cars" in p:
        continue
    else:
        folder_path = root_folder + "/" + p
        print("Updating images for " + p)
        update_images(dir_path=folder_path, cur_folder=None)
        print("Updating measurments for " + p)
        update_measurements(folder_path)