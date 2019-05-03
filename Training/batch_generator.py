import os
import pandas as pd
import numpy as np
from data_configuration import Config
from data_handler_temporal import DataHandler
from math import ceil
# TODO: FINISH

class KerasBatchGenerator(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.current_idx = 0
        self.current_recording = 0

        self.conf = Config()

        self.data = None
        self.data_paths = []

        self.folder_index = -1 #Adds before fetching
        self.measures = []
        for key in self.conf.available_columns:
            if self.conf.input_data[key]:
                if key == "Direction":
                    self.measures.append("Directions")
                else:
                    self.measures.append(key)
        
        self.fetch_folders()
        self.get_new_measurments_recording()

    def get_number_of_steps_per_epoch(self):
        """Returns total number of samples"""
        number_of_samples = 0
        self.store = pd.HDFStore("../Training_data/store.h5")

        for path in self.data_paths:
            df_name = "Recording_" + path.split("/")[-1]
            recording = self.store[df_name]
            number_of_samples += len(recording.index)
        
        self.store.close()
        return ceil(number_of_samples / self.conf.train_conf.batch_size)

    # Fetch the folders available
    def fetch_folders(self):
        for folder in sorted(os.listdir('../Training_data')):
            if folder == ".DS_Store" or folder == "store.h5":
                continue
            self.data_paths.append("../Training_data/" + folder)
        self.data_paths.sort(key=lambda a: int(a.split("/")[-1]))

    # Add measurments from the next recording
    def get_new_measurments_recording(self):
        self.store = pd.HDFStore("../Training_data/store.h5")

        self.folder_index += 1
        if self.folder_index >= len(self.data_paths):
            self.folder_index = 0
        df_name = "Recording_" + self.data_paths[self.folder_index].split("/")[-1]
        self.data = self.store[df_name]
        self.store.close()
        self.add_image_sequences()

    # Fetch image sequences for the current recording
    def add_image_sequences(self):
        self.data["Sequence"] = None
        for index, row in self.data.iterrows():
            cur_frame = row["frame"]
            sequence = np.load(self.data_paths[self.folder_index] + "/Sequences/" + str(cur_frame) + ".npy")
            self.data.at[index, "Sequence"] = np.array(sequence)

    def generate(self):
        while True:
            x = [[],[]]
            y = [] #np.zeros((self.batch_size, self.conf.input_size_data["Output"] ))
            for _ in range(self.batch_size):
                if self.current_idx + 1 >= len(self.data.index):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                    self.get_new_measurments_recording()
            
                sequence = self.get_sequence()
                measurements = self.get_measurements()
                x[0].append(sequence)
                x[1].append(measurements["Directions"])
                #x.append(self.reshape_input(sequence, measurements))
                y.append(self.get_output())
                self.current_idx += 1
            yield {"input_1": np.array(x[0]), "input_2": np.array(x[1])}, {"output": np.array(y)}

    def get_sequence(self):
        return self.data.loc[self.current_idx, "Sequence"]

    # TODO: Add preprosessing if necesarry   
    def get_measurements(self):     
        return self.data.loc[self.current_idx, self.measures]

    # TODO: Verify that input comes in the right order
    def reshape_input(self, sequence, measurements):
        x = {}
        x["input_1"] = sequence
        x["input_2"] = measurements["Directions"]
        return x
        """
        x = []
        x.append(np.array(sequence))
        for col in measurements:
            x.append(col)
        return x
        """

    def get_output(self):
        return self.data.loc[self.current_idx, "Steer"]

    @staticmethod
    def get_values_as_numpy_arrays(values):
        new_shape = values.shape + values[0].shape
        new_values = []
        for value in values:
            new_values.append(value)
        
        return np.array(new_values).reshape(new_shape)