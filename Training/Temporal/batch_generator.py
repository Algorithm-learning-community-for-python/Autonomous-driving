""" Used to list all folders"""
import os
from math import ceil
import pandas as pd
import numpy as np
from Misc.data_configuration import Config

class BatchGenerator(object):
    """ Generator that yields batches of training data concisting of multiple inputs"""
    #pylint: disable=too-many-instance-attributes
    def __init__(self):
        self.conf = Config()
        self.batch_size = self.conf.train_conf.batch_size

        self.data = None
        self.store = None
        self.data_paths = []
        self.measures = [key for key in self.conf.available_columns if self.conf.input_data[key]]
        self.current_idx = 0
        self.current_recording = 0
        self.folder_index = -1
        self.fetch_folders()

    def get_number_of_steps_per_epoch(self):
        """Returns steps per epoch = total_nr_samples / batch_size"""
        number_of_samples = 0
        store = pd.HDFStore("../../Training_data/store.h5")
        for path in self.data_paths:
            df_name = "Recording_" + path.split("/")[-1]
            recording = store[df_name]
            number_of_samples += len(recording.index)
        store.close()
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
        self.store = pd.HDFStore("../../Training_data/store.h5")
        self.folder_index += 1
        if self.folder_index >= len(self.data_paths):
            self.folder_index = 0
        df_name = "Recording_" + self.data_paths[self.folder_index].split("/")[-1]
        self.data = self.store[df_name]
        self.store.close()
        self.add_image_sequences()

    def add_image_sequences(self):
        """Fetches image sequences for the current recording"""
        self.data["Sequence"] = None
        for index, row in self.data.iterrows():
            cur_frame = row["frame"]
            sequence = np.load(
                self.data_paths[self.folder_index] + "/Sequences/" + str(cur_frame) + ".npy"
            )
            self.data.at[index, "Sequence"] = np.array(sequence)

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
                x[0].append(self.get_sequence())
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

    def get_sequence(self):
        """" Returns the image sequence"""
        return self.data.loc[self.current_idx, "Sequence"]

    def get_measurements(self):
        """ Returns the measurments"""
        return self.data.loc[self.current_idx, self.measures]

    def get_output(self):
        """ Returns the output, currently only STEER"""
        return self.data.loc[self.current_idx, "Steer"][-1]
