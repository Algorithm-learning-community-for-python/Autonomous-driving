"""Spatial generator"""
from math import ceil
import random
import pandas as pd
import numpy as np
from Misc.misc import get_image, get_data_paths
from Misc.preprocessing import filter_input_based_on_steering

class BatchGenerator(object):
    """ Generator that yields batches of training data concisting of multiple inputs"""
    #pylint: disable=too-many-instance-attributes
    def __init__(self, conf, data="Training_data"):
        self.conf = conf
        self.batch_size = self.conf.train_conf.batch_size
        self.data = None
        self.data_paths = get_data_paths(data)
        self.input_measures = [
            key for key in self.conf.available_columns if self.conf.input_data[key]
            ]
        self.output_measures = [
            key for key in self.conf.available_columns if self.conf.output_data[key]
            ]
        self.current_idx = 0

        self.folder_index = -1




    def get_number_of_steps_per_epoch(self):
        """Returns steps per epoch = total_nr_samples / batch_size"""
        number_of_samples = 0
        for path in self.data_paths:
            data = pd.read_csv(path + self.conf.recordings_path)
            number_of_samples += len(data.index)
        return ceil(number_of_samples / self.conf.train_conf.batch_size)

    def get_new_measurments_recording(self):
        """Loads measurments from the next recording"""
        self.folder_index += 1
        if self.folder_index >= len(self.data_paths):
            self.folder_index = 0

        path = self.data_paths[self.folder_index] + self.conf.recordings_path
        self.data = pd.read_csv(path)

        #Filter
        if self.conf.filter_input:
            self.data = filter_input_based_on_steering(self.data, self.conf, temporal=False)

        for index, row in self.data.iterrows():
            self.data.at[index, "Direction"] = [int(x) for x in row["Direction"].strip("][").split(".")[:-1]]
            self.data.at[index, "TL_state"] = [int(x) for x in row["TL_state"].strip("][").split(".")[:-1]]
        #self.add_images()
    """
    def add_images(self):
        """"""Fetches image sequences for the current recording""""""
        self.data["Image"] = None
        path = self.data_paths[self.folder_index] + "/Images/"
        for index, row in self.data.iterrows():
            frame = str(row["frame"])
            img = get_image(path, frame)
            img = img[..., ::-1]
            img = img[self.conf.top_crop:, :, :]
            self.data.at[index, "Image"] = np.array(img)
    """
    def generate(self):
        """ Yields images and chosen control signals(measures) in the size of batches"""
        #pylint: disable=invalid-name
        self.folder_index = random.randint(-1, len(self.data_paths)-1)
        self.get_new_measurments_recording() # Fetches the initial measurment recording
        while True:
            x = [[]]
            for i in range(len(self.input_measures)):
                x.append([])
            y = []
            for i in range(len(self.output_measures)):
                y.append([])

            for _ in range(self.batch_size):
                # Check if a new recording should be fetched
                if self.current_idx + 1 >= len(self.data.index):
                    self.current_idx = 0
                    self.get_new_measurments_recording()

                # Add the current sequence to the batch
                x[0].append(self.get_image())
                input_measurements = self.get_measurements(self.input_measures)
                output_measurements = self.get_measurements(self.output_measures)

                for i, measure in enumerate(self.input_measures):
                    x[i + 1].append(input_measurements[measure])

                for i, measure in enumerate(self.output_measures):
                    y[i].append(output_measurements[measure])

                self.current_idx += 1

            # Convert x to dict to allow for multiple inputs
            X = {"input_Image": np.array(x[0])}
            Y = {}
            for i, measure in enumerate(self.input_measures):
                X["input_" + measure] = np.array(x[i+1])

            for i, measure in enumerate(self.output_measures):
                Y["output_" + measure] = np.array(y[i])
            yield X, Y
            #print("Yielded a batch from recording " + str(self.folder_index) + ". cur_idx=" + str(self.current_idx))

    def get_image(self):
        """" Returns the image sequence"""
        path = self.data_paths[self.folder_index] + "/Images/"
        frame = str(self.data.loc[self.current_idx, "frame"])
        img = get_image(path, frame)
        img = img[..., ::-1]
        img = img[self.conf.top_crop:, :, :]
        return img

    def get_measurements(self, measures):
        """ Returns the measurments"""
        return self.data.loc[self.current_idx, measures]

    def get_output(self):
        """ Returns the output, currently only STEER"""
        return self.data.loc[self.current_idx, "Steer"]
