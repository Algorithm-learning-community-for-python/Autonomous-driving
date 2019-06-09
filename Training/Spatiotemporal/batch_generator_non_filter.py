"""Spatiotemporal generator"""
from math import ceil
import random
import pandas as pd
import numpy as np
from keras.utils import Sequence
from Misc.misc import get_image, get_data_paths
from Misc.preprocessing import filter_input_based_on_steering, filter_input_based_on_speed_and_tl, filter_corrupt_input

class BatchGenerator(Sequence):
    """ Generator that yields batches of training data concisting of multiple inputs"""
    #pylint: disable=too-many-instance-attributes
    def __init__(self, conf, data="Training_data"):
        self.conf = conf
        self.batch_size = self.conf.train_conf.batch_size
        self.seq_len = self.conf.input_size_data["Sequence_length"]
        self.data = None
        self.data_paths = get_data_paths(data)
        self.input_measures = [
            key for key in self.conf.available_columns if self.conf.input_data[key]
            ]
        self.output_measures = [
            key for key in self.conf.available_columns if self.conf.output_data[key]
            ]
        self.get_measurements_recordings(data)

    def __len__(self):
        return np.ceil(len(self.data.index)/self.batch_size)

    def __getitem__(self, idx):
        images = []
        measurments =  []

        for i in range(self.seq_len):
            images.append([])
            measurments.append([])
            for j in range(len(self.input_measures)):
                measurments[i].append([])

        y = []
        for i in range(len(self.output_measures)):
            y.append([])

        cur_idx = idx*self.batch_size
        shift = 0
        for _ in range(self.batch_size):
            #print("new_sample")
            # Add the current sequence to the batch
            initial_row = self.data.iloc[cur_idx, :]
            last_row = self.data.iloc[cur_idx + (self.seq_len-1), :]
            if initial_row["Recording"] != last_row["Recording"]:
                shift = self.seq_len  - 1

            for j in range(self.seq_len):
                row = self.data.iloc[cur_idx + shift + j, :]
                #print(self.data_paths[row["Recording"]])
                #print(str(row["frame"]))
                images[j].append(self.get_image(row))
                input_measurements = row[self.input_measures]
                output_measurements = row[self.output_measures]
                    

                for i, measure in enumerate(self.input_measures):
                    measurments[j][i].append(input_measurements[measure])

            for i, measure in enumerate(self.output_measures):
                y[i].append(output_measurements[measure])
            cur_idx += 1

        # Convert x to dict to allow for multiple inputs
        X = {}
        for j in range(self.seq_len): 
            X["input_Image"+str(j)] = np.array(images[j])
            for i, measure in enumerate(self.input_measures):
                X["input_" + measure + str(j)] = np.array(measurments[j][i])

        Y = {}
        for i, measure in enumerate(self.output_measures):
            Y["output_" + measure] = np.array(y[i])
        return X, Y

    def get_measurements_recordings(self, data):
        dfs = []
        for i, path in enumerate(self.data_paths):
            df = pd.read_csv(path + self.conf.recordings_path)
            df["Recording"] = i 
            dfs.append(df)
        self.data = pd.concat(dfs, ignore_index=True)

        #Filter
        if self.conf.filter_input and data != "Validation_data":
            self.data = filter_input_based_on_steering(self.data, self.conf, temporal=False)
            self.data = filter_input_based_on_speed_and_tl(self.data, self.conf, temporal=False)
            self.data = filter_corrupt_input(self.data, self.conf, temporal=False)

        for index, row in self.data.iterrows():
            if self.conf.input_data["Direction"]:
                self.data.at[index, "Direction"] = [int(x) for x in row["Direction"].strip("][").split(".")[:-1]]
            if self.conf.input_data["TL_state"]:
                self.data.at[index, "TL_state"] = [int(x) for x in row["TL_state"].strip("][").split(".")[:-1]]
            if self.conf.input_data["ohe_speed_limit"]:
                self.data.at[index, "ohe_speed_limit"] = [int(x) for x in row["ohe_speed_limit"].strip("][").split(".")[:-1]]


    def get_image(self, row):
        """" Returns the image sequence"""
        path = self.data_paths[row["Recording"]] + "/Images/"
        frame = str(row["frame"])
        img = get_image(path, frame)

        img = img[..., ::-1]
        img = img[self.conf.top_crop:, :, :]
        return img
