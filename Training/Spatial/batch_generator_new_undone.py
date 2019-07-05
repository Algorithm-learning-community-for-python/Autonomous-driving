"""Spatial generator"""
#pylint: disable=too-many-instance-attributes
#pylint: disable=line-to-long
#pylint: disable=undefined-variable
#pylint: disable=no-member
from __future__ import print_function
import random
import pickle
import pandas as pd
import numpy as np
import cv2

from keras.utils import Sequence
from Misc.misc import get_image, get_data_paths
from Misc.preprocessing import (
    filter_input_based_on_steering, \
    filter_input_based_on_speed_and_tl, \
    filter_corrupt_input, \
)

class BatchGenerator(Sequence):
    """ Generator that yields batches of training data concisting of multiple inputs"""
    #pylint: disable=too-many-instance-attributes
    def __init__(self, conf, data="Training_data"):
        self.conf = conf
        self.img_size = self.conf.input_size_data["Image"]
        self.batch_size = self.conf.train_conf.batch_size
        self.data = None
        self.data_type = data
        self.data_paths = []
        # Used to store set of path_index, sample_index
        self.indexes = []
        self.count_samples = 0

        print("Fetching folders")
        if data == "Training_data":
            for dataset in conf.data_paths:
                self.data_paths.extend(get_data_paths(data + "/" + dataset))
        else:
            for dataset in conf.data_paths_validation_data:
                self.data_paths.extend(get_data_paths(data + "/" + dataset))
        print("Fetched " + str(len(self.data_paths)) + " episodes")

        try:
            with open ('filtered_indexes_' + data, 'rb') as fp:
                print("Found file containing filtered indexes, using these")
                self.indexes = pickle.load(fp)
            self.count_samples = len(self.indexes)
            print("Fetched " + str(self.count_samples) + " samples from file")
        except FileNotFoundError as fe:
            print("No file found, creating index file")
            self.get_indexes()
            print("Storing index file for later use...")
            with open('filtered_indexes_' + data, 'wb') as fp:
                pickle.dump(self.indexes, fp)

        self.input_measures = [
            key for key in self.conf.available_columns if self.conf.input_data[key]
            ]
        self.output_measures = [
            key for key in self.conf.available_columns if self.conf.output_data[key]
            ]

    def __len__(self):
        return int(np.floor(self.count_samples/self.batch_size))

    def __getitem__(self, idx):
        # Shuffle randomly from the validation set
        if self.data_type == "Validation_data":
            cur_idx = random.randint(0, len(self.indexes)-1)
        else:
            cur_idx = idx*self.batch_size

        x = [[]]
        for i in range(len(self.input_measures)):
            x.append([])
        y = []
        for i in range(len(self.output_measures)):
            y.append([])
        
        batch = self.get_batch_of_measurement_recordings(cur_idx)
        for b in range(self.batch_size):
            # Add the current sequence to the batch
            row = batch[b]

            x[0].append(self.get_image(row))
            input_measurements = row[self.input_measures]
            output_measurements = row[self.output_measures]
                

            for i, measure in enumerate(self.input_measures):
                x[i + 1].append(input_measurements[measure])

            for i, measure in enumerate(self.output_measures):
                y[i].append(output_measurements[measure])

        # Convert x to dict to allow for multiple inputs
        X = {"input_Image": np.array(x[0])}
        Y = {}
        for i, measure in enumerate(self.input_measures):
            X["input_" + measure] = np.array(x[i+1])

        for i, measure in enumerate(self.output_measures):
            Y["output_" + measure] = np.array(y[i])
        return X, Y

    def get_batch_of_measurement_recordings(self, cur_idx):
        """ 
        Fetches one batch of measurments. 
        """ 
        path_idx, _ = self.indexes[cur_idx]
        df = pd.read_csv(self.data_paths[path_idx] + self.conf.recordings_path)
        df["Images_path"] = self.data_paths[path_idx] + self.conf.images_path
        batch = []
        l = len(self.indexes)
        exception_idx = 0

        # Add samples to the batch
        for i in range(self.batch_size):
            #Check if we have reached the end of indexes
            if cur_idx + i >= l:
                cur_path_idx, sequence_idx = self.indexes[exception_idx]
                exception_idx += 1
            else:
                cur_path_idx, sequence_idx = self.indexes[cur_idx + i]

            #Check if we have reached the end of the current dataframe
            if path_idx != cur_path_idx:
                df = pd.read_csv(self.data_paths[cur_path_idx] + self.conf.recordings_path)
                df["Images_path"] = self.data_paths[cur_path_idx] + self.conf.images_path
                path_idx = cur_path_idx

            #Add the sample
            batch.append(df.iloc[sequence_idx, :])
            
        #Convert string data to arrays
        for row in batch:
            if self.conf.input_data["Direction"]:
                row["Direction"] = [int(x) for x in row["Direction"].strip("][").split(".")[:-1]]
            if self.conf.input_data["TL_state"]:
                row["TL_state"] = [int(x) for x in row["TL_state"].strip("][").split(".")[:-1]]
            if self.conf.input_data["ohe_speed_limit"]:
                row["ohe_speed_limit"] = [int(x) for x in row["ohe_speed_limit"].strip("][").split(".")[:-1]]

        return batch

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


    def get_indexes(self):
        skip_steps = self.conf.skip_steps
        count_filtered = 0
        count_samples = 0
        for p, path in enumerate(self.data_paths):
            print("\r fetching indexes from path number " + str(p), end="")
            df = pd.read_csv(path + self.conf.recordings_path)
            l = len(df.index)
            
            # TODO: FIX IF NECESARRY
            for i in range(0, l, skip_steps):
                # Create a sequence
                indexes = []
                for sample_idx in range(i, i + (self.seq_len*step_size), step_size):
                    indexes.append(sample_idx)
                temp_sequence = df.iloc[indexes, :].copy()
                
                # Filter away sequence based on conditions
                if self.conf.filter_input and self.data_type != "Validation_data":
                    if filter_one_sequence_based_on_steering(temp_sequence, self.conf) or filter_one_sequence_based_on_not_moving(temp_sequence, self.conf):
                        count_filtered += 1
                        continue
                count_samples += 1
                self.indexes.append((p,i))
        self.count_samples = count_samples
        print("\n Filtered out " + str(count_filtered) + " out of " + str(count_samples + count_filtered))
