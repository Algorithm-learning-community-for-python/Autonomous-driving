"""Updates the measurment file """
from __future__ import print_function


import os, sys
import pandas as pd
import numpy as np
import cv2
from Misc.misc import get_data_paths, get_image_name, get_image
from Misc.preprocessing import get_one_hot_encoded
from Spatial.data_configuration import Config

def test_image_resize(path):
    CONF = Config()
    MEASURMENT_PATH = "/Measurments/recording.csv"
    DATA_PATHS = []
    #PATH = '../../Training_data'
    #PATH = '../../Validation_data'
    PATH = "../../" + path 
    cur_folder = 0
    if cur_folder == None:
        for folder in os.listdir(PATH):
            if folder == "store.h5":
                continue
            DATA_PATHS.append(PATH + "/" + folder)
    else:
        DATA_PATHS.append(PATH + "/" + str(cur_folder))

    for path in DATA_PATHS:
        print(path)
        df = pd.read_csv(path + MEASURMENT_PATH)
        img_array = []
        hsv_array = []
        for j, row in df.iterrows():
            cur_frame = row["frame"]
            img = get_image(path + "/Images/", cur_frame)
            img_size = CONF.input_size_data["Image"]
            img = img[CONF.top_crop:, :, :]
            img = img[..., ::-1]
            cv2.imshow("image", img)
            cv2.waitKey(0)
            img = cv2.resize(img,(img_size[1], img_size[0]))
            cv2.imshow("image", img)
            cv2.waitKey(0)

def move_training_data(path):
    DATA_PATHS = get_data_paths(path)
    start_index = 0
    end_index = 45
    new_index = 0
    for path in DATA_PATHS[start_index:]:

        # listing directories
        print("The dir is: " + path)

        # renaming directory ''tutorialsdir"
        new_path = "../../Training_data_no_cars_with_noise/" + str(new_index)
        print("The new dir is: " + new_path)
        os.rename(path, new_path)

        print("Successfully renamed.")
        new_index += 1

def get_average_step_size(path):
    DATA_PATHS = get_data_paths(path)
    sum_steps = 0
    total_samples = 0
    for path in DATA_PATHS:

        images = get_data_paths(path[6:] + "/Images/", sort=False)
        df = pd.read_csv(path + "/Measurments/recording.csv")

        # Check if there is a jump in timeframes
        df_sum = 0
        df_samples = 0
        previous = 0
        for i, row in df.iterrows():
            total_samples += 1
            df_samples += 1
            current = int(row["frame"])
            if previous != 0:
                sum_steps += current - previous
                df_sum += current - previous
            previous = current
        print(path)
        print("Average step_size: " + str(float(df_sum)/df_samples))
    print("Total averag step size for all episodes = " + str(float(sum_steps)/total_samples))
    
def find_empty_folders(path):
    DATA_PATHS = get_data_paths()
    for path in DATA_PATHS:
        images = get_data_paths(path[6:] + "/Images/", sort=False)
        df = pd.read_csv(path + "/Measurments/recording.csv")

        if len(images) != len(df.index) or len(images) < 50:
            print(path)
            print("Images: " + str(len(images)))
            print("Measurments: " + str(len(df.index)))

    
path = "Training_data_temp"
#find_empty_folders(path)
#get_average_step_size(path)
#find_empty_folders(path)
#test_image_resize(path)
move_training_data(path)