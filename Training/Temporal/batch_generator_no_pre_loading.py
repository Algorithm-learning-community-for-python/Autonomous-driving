"""
Temporal generator
NB: Loads duplicate samples
"""
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
                df = pd.read_csv(folder + self.conf.recordings_path)
                self.samples_per_data_path.append(len(df.index))
                self.data_paths.append(folder)
        print("Fetched " + str(len(self.data_paths)) + " episodes")

        self.input_measures = [
            key for key in self.conf.available_columns if self.conf.input_data[key]
            ]
        self.output_measures = [
            key for key in self.conf.available_columns if self.conf.output_data[key]
            ]
        self.get_measurements_recordings(data)
        self.X = {
            "input_Image": np.zeros([self.batch_size, self.seq_len] + self.conf.input_size_data["Image"]),
            "input_Direction": np.zeros([self.batch_size, self.seq_len] + self.conf.input_size_data["Direction"]),
            "input_Speed": np.zeros([self.batch_size, self.seq_len] + self.conf.input_size_data["Speed"]),
            #"input_frame": np.zeros([self.batch_size, self.seq_len] + self.conf.input_size_data["frame"]),
            "input_ohe_speed_limit": np.zeros([self.batch_size, self.seq_len] + self.conf.input_size_data["ohe_speed_limit"]),
            "input_TL_state": np.zeros([self.batch_size, self.seq_len] + self.conf.input_size_data["TL_state"]) 
        }
        self.Y = {
            "output_Throttle": np.zeros([self.batch_size, 1]),
            "output_Brake": np.zeros([self.batch_size, 1]),
            "output_Steer": np.zeros([self.batch_size, 1]),
        }

    def __len__(self):
        total_samples = sum(self.samples_per_data_path)
        total_samples = total_samples - ((self.seq_len*self.conf.step_size_training)*len(self.samples_per_data_path))
        total_batches = int(np.floor(total_samples/self.batch_size))
        return total_batches


    def __getitem__(self, idx):
        measurments = self.get_batch_of_measurement_recordings(idx*self.batch_size)
        for b, sequence in enumerate(measurments):
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
        return self.X, self.Y

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

    def get_batch_of_measurement_recordings(self, cur_idx):
        """ 
        Fetches one batch of measurments. 
        Due to filtering, it has a high probabillity of fetching the same samples.
        Skip_steps can be set to a higher value to lower this probabillity
        """ 

        step_size = self.conf.step_size_training
        skip_steps = self.conf.skip_steps
        batch = []
        skipped_samples = 0
        idx = cur_idx
        df_idx = 0
        path_idx = 0
        count_filtered = 0
        # Find the corresponding path_idx and sample_idx to the current idx
        path = self.data_paths[0]
        samples = self.samples_per_data_path[0]
        while idx >= samples + (self.seq_len*step_size):
            idx -= samples
            path_idx += 1
            # if we have reached the end of paths, then start over again
            if path_idx >= len(self.data_paths):
                print("Starting over again with episodes")
                path_idx = 0
            samples = self.samples_per_data_path[path_idx]


                
        # Once the correct recording is found, we start to add the batch
        # idx now corresponds to the index of the first sample in the current path
        path = self.data_paths[path_idx]
        df = pd.read_csv(path + self.conf.recordings_path)
        #print("fetching from " + path)
        l = len(df.index)
        seq_idx = idx
        # Iterate until the batch is filled up
        while len(batch) < self.batch_size:
            # Fetch next df if the end of current is reached
            if seq_idx + (self.seq_len*step_size) >= l:
                print("reached end of df, fetching from next df")
                df_idx += 1
                # Reset indexes if end of paths is reached
                if path_idx + df_idx >= len(self.data_paths):
                    print("Starting over again with episodes")
                    path_idx = 0
                    df_idx = 0
                path = self.data_paths[path_idx + df_idx]
                print(path)
                df = pd.read_csv(self.data_paths[path_idx + df_idx] + self.conf.recordings_path)
                l = len(df.index)
                seq_idx = 0

            # Create a sequence
            indexes = []
            for sample_idx in range(seq_idx, seq_idx + (self.seq_len*step_size), step_size):
                indexes.append(sample_idx)
            temp_sequence = df.iloc[indexes, :].copy()
            temp_sequence["Images_path"] = path + self.conf.images_path
            
            seq_idx += step_size
            # Filter away sequence based on conditions
            if self.conf.filter_input and self.data_type != "Validation_data":
                if filter_one_sequence_based_on_steering(temp_sequence, self.conf) or filter_one_sequence_based_on_not_moving(temp_sequence, self.conf):
                    count_filtered += 1
                    continue

            #Convert string data to arrays
            for index, row in temp_sequence.iterrows():
                if self.conf.input_data["Direction"]:
                    temp_sequence.at[index, "Direction"] = [int(x) for x in str(row["Direction"]).strip("][").split(".")[:-1]]
                if self.conf.input_data["TL_state"]:
                    temp_sequence.at[index, "TL_state"] = [int(x) for x in str(row["TL_state"]).strip("][").split(".")[:-1]]
                if self.conf.input_data["ohe_speed_limit"]:
                    temp_sequence.at[index, "ohe_speed_limit"] = [int(x) for x in str(row["ohe_speed_limit"]).strip("][").split(".")[:-1]]
            batch.append(temp_sequence)
        #print("filtered out " + str(count_filtered))
        return batch
