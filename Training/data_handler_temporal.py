# Div
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
import numpy as np
import scipy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random

# TODO: delete and replace with generator


class DataHandler:
    def __init__(self, atrX=None, atrY=None, train_valid_split=0.2):
        
        self.data = None                # Contains all data 
        self.data_as_recordings = []     # Contains all data divided into recordings

        self.dataX = None               # Contains all rows of the collums Defined by atrX
        self.dataY = None               # Contains all rows of the collums Defined by atrY

        self.training_data = None       # Contains random subset of rows defined by train_valid_split, has all attributes
        self.validation_data = None     # Contains random  subset of rows defined by train_valid_split, has all attributes

        self.trainX = None              # Contains random subset of rows defined by train_valid_split and the collums Defined by atrX
        self.validX = None              # Contains random subset of rows defined by train_valid_split and the collums Defined by atrX

        self.trainY = None              # Contains random subset of rows defined by train_valid_split and the collums Defined by atrY
        self.validY = None              # Contains random subset of rows defined by train_valid_split and the collums Defined by atrY
        self.bottom_crop = 115
        self.top_crop = 100 

        self.data_paths = []
        self.fetch_data()

    def fetch_data(self):
        print("fetching measurments")
        self.fetch_measurements()
        print("fetching sequences")
        self.fetch_sequences()
        print("concatinating")
        self.data = pd.concat(self.data_as_recordings, ignore_index=True)


    def fetch_measurements(self):
        folders = []
        for folder in os.listdir('../Training_data'):
            if folder == ".DS_Store" or folder == "store.h5":
                continue
            self.data_paths.append("../Training_data/" + folder)
            folders.append(folder)

        store = pd.HDFStore("../Training_data/store.h5")
        for folder in folders:
            df = store[folder]
            self.data_as_recordings.append(df)
        store.close()

    def fetch_sequences(self):
        for j, measurement_recording in enumerate(self.data_as_recordings):
            measurement_recording["Sequence"] = None
            for index, row in measurement_recording.iterrows():
                cur_frame = row["frame"]
                sequence = np.load(self.data_paths[j] + "/Sequences/" + str(cur_frame) + ".npy")
                measurement_recording.at[index, "Sequence"] = np.array(sequence)

    def set_train_valid_split(self, test_size=0.25): 
        data = self.get_data(as_one_list=True)
        split = train_test_split(data, test_size=test_size)
        (self.training_data, self.validation_data) = split

    
    def set_XY_data(self, atrX, atrY, train_valid_split=True):
        if train_valid_split:
            self.trainX, self.validX = self.get_attributes(atrX)
            self.trainY, self.validY = self.get_attributes(atrY)
        else: 
            self.dataX = self.get_attributes(atrX, train_valid_split=False)
            self.dataY = self.get_attributes(atrY, train_valid_split=False)
    
    def get_attributes(self, atr, train_valid_split=True):
        if train_valid_split:
            train = self.training_data.loc[:,atr]
            valid = self.validation_data.loc[:,atr]
            return train, valid
        else: 
            data = self.get_data(as_one_list=True)
            return data.loc[:,atr]

    def get_data(self, as_one_list=True):
        if as_one_list:
            return self.data
        else:
            return self.data_as_recordings

    @staticmethod
    def get_values_as_numpy_arrays(values):
        print(values)
        print(values.shape)
        new_shape = values.shape + values[0].shape
        new_values = []
        for value in values:
            new_values.append(value)
        
        return np.array(new_values).reshape(new_shape)

    def plot_data(self, data=None, images=None):
        if data is None:
            data = self.get_data(as_one_list=True)
        if images is None:
            images = data.Sequence.values

        fig = plt.figure(figsize=(8, 8))

        columns = 5
        rows = 4

        for i in range(1, columns * rows + 1):
            j = (i-1) % 5
            if j == 0:
                r = np.random.randint(len(images))
                sequence = images[r]
                steering = "steering: " + str(round(float(data.loc[r, "Steer"]), 3))
                direction = "direction: " + str(data.loc[r, "Direction"])

                fig.add_subplot(rows, columns, i, title=(steering + " - " + direction))
                plt.imshow(sequence[j])
            else:
                fig.add_subplot(rows, columns, i)
                plt.imshow(sequence[j])
        plt.savefig('testfigure.png')
