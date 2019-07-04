"""Temporal generator"""
from __future__ import print_function
from math import ceil
import random
import pandas as pd
import numpy as np
import cv2
import pickle

from keras.utils import Sequence
from Misc.misc import get_image, get_data_paths
from Misc.preprocessing import (
    filter_sequence_input_based_on_steering, \
    filter_sequence_input_based_on_not_moving, \
    filter_corrupt_sequence_input, \
    filter_one_sequence_based_on_steering, \
    filter_one_sequence_based_on_not_moving, \
    augment_image
)

class BatchGenerator(Sequence):
    """ Generator that yields batches of training data concisting of multiple inputs"""
    #pylint: disable=too-many-instance-attributes
    def __init__(self, conf, data="Training_data"):
        self.conf = conf
        self.img_size = self.conf.input_size_data["Image"]
        self.batch_size = self.conf.train_conf.batch_size
        self.seq_len = self.conf.input_size_data["Sequence_length"]
        self.data = None
        self.data_type = data
        self.data_paths = []
        self.samples_per_data_path = []
        print("Fetching folders")
        for dataset in conf.data_paths:
            for folder in get_data_paths(data + "/" + dataset):
                self.data_paths.append(folder)
        print("Fetched " + str(len(self.data_paths)) + " episodes")
            
        self.input_measures = [
            key for key in self.conf.available_columns if self.conf.input_data[key]
            ]
        self.output_measures = [
            key for key in self.conf.available_columns if self.conf.output_data[key]
            ]
        #self.get_measurements_recordings(data)
        self.X = {
            "input_Image": np.zeros([self.batch_size, self.seq_len] + self.conf.input_size_data["Image"]),
            "input_Direction": np.zeros([self.batch_size, self.seq_len] + self.conf.input_size_data["Direction"]),
            "input_Speed": np.zeros([self.batch_size, self.seq_len] + self.conf.input_size_data["Speed"]),
            "input_ohe_speed_limit": np.zeros([self.batch_size, self.seq_len] + self.conf.input_size_data["ohe_speed_limit"]),
            "input_TL_state": np.zeros([self.batch_size, self.seq_len] + self.conf.input_size_data["TL_state"]) 
        }
        self.Y = {
            "output_Throttle": np.zeros([self.batch_size, 1]),
            "output_Brake": np.zeros([self.batch_size, 1]),
            "output_Steer": np.zeros([self.batch_size, 1]),
        }

    def __len__(self):
        return int(np.floor(len(self.data)/self.batch_size))

    def __getitem__(self, idx):
        cur_idx = idx*self.batch_size
        for b in range(self.batch_size):
            #Set Input
            sequence = self.data[cur_idx]
            for j in range(self.seq_len):
                current_row = sequence.iloc[j, :]
                self.X["input_Image"][b, j, :, :, :] = self.get_image(current_row)
                for measure in self.input_measures:
                    if measure == "Speed":
                        self.X["input_" + measure][b, j, :] = [current_row[measure]]
                    else:
                        self.X["input_" + measure][b, j, :] = current_row[measure]
            # Set target
            target_row = sequence.iloc[-1, :]
            for measure in self.output_measures:
                self.Y["output_" + measure][b, :] = target_row[measure]
            cur_idx += 1

        return self.X, self.Y

    def get_measurements_recordings(self, data):
        self.data = []
        step_size = self.conf.step_size_training
        skip_steps = self.conf.skip_steps
        skipped_samples = 0
        # Use subset avoid using all the data
        if data == "Validation_data":
            percentage_of_training_data = 1
        else:
            percentage_of_training_data = 1
        subset = int(len(self.data_paths)*percentage_of_training_data)
        for path in self.data_paths[:subset]:
            df = pd.read_csv(path + self.conf.recordings_path)
            df["Images_path"] = path + self.conf.images_path
            for i in range(0, len(df)):
                if i + (self.seq_len*step_size) < len(df):
                    indexes = []
                    lanefollow = True
                    for j in range(i, i + (self.seq_len*step_size), step_size):
                        if df.iloc[j, :].Direction != "[0. 0. 1. 0. 0. 0. 0.]" and df.iloc[j, :].Direction != "[0. 0. 0. 0. 0. 0. 1.]" :
                            lanefollow = False
                        indexes.append(j)
                    if i % skip_steps == 0:
                        self.data.append(df.iloc[indexes, :].copy())
                    elif not lanefollow:
                        self.data.append(df.iloc[indexes, :].copy())
                    else:
                        skipped_samples += 1
        #Filter
        if self.conf.filter_input and data != "Validation_data":
            self.data = filter_sequence_input_based_on_steering(self.data, self.conf)
            self.data = filter_sequence_input_based_on_not_moving(self.data, self.conf)
            self.data = filter_corrupt_sequence_input(self.data)
        #Convert string data to arrays
        for i, sequence in enumerate(self.data):
            for index, row in sequence.iterrows():
                if self.conf.input_data["Direction"]:
                    self.data[i].at[index, "Direction"] = [int(x) for x in str(row["Direction"]).strip("][").split(".")[:-1]]
                if self.conf.input_data["TL_state"]:
                    self.data[i].at[index, "TL_state"] = [int(x) for x in str(row["TL_state"]).strip("][").split(".")[:-1]]
                if self.conf.input_data["ohe_speed_limit"]:
                    self.data[i].at[index, "ohe_speed_limit"] = [int(x) for x in str(row["ohe_speed_limit"]).strip("][").split(".")[:-1]]       

    def get_image(self, row):
        """" Returns the image corresponding to the row"""
        path = row["Images_path"]
        frame = str(row["frame"])
        img = get_image(path, frame)
        if self.conf.images_path == "/Images/":
            img = img[self.conf.top_crop:, :, :]
            img = cv2.resize(img,(self.img_size[1], self.img_size[0]))
        img = img[..., ::-1]
        return img
