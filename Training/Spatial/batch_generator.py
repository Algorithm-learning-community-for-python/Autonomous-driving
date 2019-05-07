"""Spatial generator"""
import os
from math import ceil
import random
import pandas as pd
import numpy as np
import cv2

from Misc.preprocessing import get_one_hot_encoded

class BatchGenerator(object):
    """ Generator that yields batches of training data concisting of multiple inputs"""
    #pylint: disable=too-many-instance-attributes
    def __init__(self, conf):
        self.conf = conf
        self.batch_size = self.conf.train_conf.batch_size

        self.data = None
        self.data_paths = []
        self.measures = [key for key in self.conf.available_columns if self.conf.input_data[key]]
        self.current_idx = 0
        self.current_recording = 0
        self.folder_index = -1
        self.fetch_folders()

    def get_number_of_steps_per_epoch(self):
        """Returns steps per epoch = total_nr_samples / batch_size"""
        number_of_samples = 0
        for path in self.data_paths:
            data = pd.read_csv(path + "/Measurments/recording.csv")
            number_of_samples += len(data.index)
        return ceil(number_of_samples / self.conf.train_conf.batch_size)


    def fetch_folders(self):
        """ Fetch all available folders """
        for folder in sorted(os.listdir('../../Training_data')):
            if folder == ".DS_Store" or folder == "store.h5":
                continue
            self.data_paths.append("../../Training_data/" + folder)
        self.data_paths.sort(key=lambda a: int(a.split("/")[-1]))


    def get_new_measurments_recording(self):
        """Loads measurments from the next recording"""
        self.folder_index += 1
        if self.folder_index >= len(self.data_paths):
            self.folder_index = 0

        path = self.data_paths[self.folder_index] + "/Measurments/recording.csv"
        self.data = pd.read_csv(path)

        #Filter
        self.data = self.data.drop(
            self.data[(np.power(self.data.Steer, 2) < 0.001) & \
            random.randint(0, 10) > (10 - (10 * self.conf.filtering_degree))].index)

        #OHE
        ohe_directions = get_one_hot_encoded(self.conf.direction_categories, self.data.Direction)
        for index, _ in self.data.iterrows():
            self.data.at[index, "Direction"] = ohe_directions[index]

        self.add_images()

    def add_images(self):
        """Fetches image sequences for the current recording"""
        self.data["Image"] = None
        for index, row in self.data.iterrows():
            l = len(str(row["frame"]))
            pad = ''
            for i in range(8 - l):
                pad += '0'
            frame = str(row["frame"])

            # Fetch image
            file_path = self.data_paths[self.folder_index] + "/Images/" + pad + frame + '.png'
            img = cv2.imread(file_path)
            if len(img) == 0:
                print("Error fetching image")

            # Converting to rgb
            img = img[..., ::-1]

            # Cropping
            img = img[self.conf.top_crop:, :,:]
            #img = cv2.resize(img,(200,88))

            # Add to recording
            self.data.at[index, "Image"] = np.array(img)

    def generate(self):
        """ Yields images and chosen control signals(measures) in the size of batches"""
        #pylint: disable=invalid-name
        self.get_new_measurments_recording() # Fetches the initial measurment recording
        while True:
            x = [[]]
            for i in range(len(self.measures)):
                x.append([])
            y = []
            for _ in range(self.batch_size):
                # Check if a new recording should be fetched
                if self.current_idx + 1 >= len(self.data.index):
                    self.current_idx = 0
                    self.get_new_measurments_recording()

                # Add the current sequence to the batch
                x[0].append(self.get_image())
                measurements = self.get_measurements()
                for i, measure in enumerate(self.measures):
                    x[i + 1].append(measurements[measure])
                y.append(self.get_output())
                self.current_idx += 1

            # Convert x to dict to allow for multiple inputs
            X = {"input_1": np.array(x[0])}
            Y = {"output": np.array(y)}
            for i, measure in enumerate(self.measures):
                X["input_" + str(i + 2)] = np.array(x[i+1])

            yield X, Y

    def get_image(self):
        """" Returns the image sequence"""
        return self.data.loc[self.current_idx, "Image"]

    def get_measurements(self):
        """ Returns the measurments"""
        return self.data.loc[self.current_idx, self.measures]

    def get_output(self):
        """ Returns the output, currently only STEER"""
        return self.data.loc[self.current_idx, "Steer"]
