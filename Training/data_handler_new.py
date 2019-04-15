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

class DataHandler():
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

        self.fetch_data()


    def fetch_data(self):
        measurements_path = "/Measurments/recording.csv"
        image_path = "/Images/"

        data_paths=[]
        for folder in os.listdir('../Training_data'):
            data_paths.append("../Training_data/" + folder)

        for path in data_paths:
            measurment_recording = pd.read_csv(path + measurements_path )
            print("Shape before filtering: ")
            print(measurment_recording.shape)
            measurment_recording = self.filter_input(measurment_recording)
            print("Shape after filtering: ")
            print(measurment_recording.shape)
            self.data_as_recordings.append(measurment_recording)

        for j, measurment_recording in enumerate(self.data_as_recordings):
            measurment_recording["Image"] = None
            for index, row in measurment_recording.iterrows():
                # Add padding to frame number
                l = len(str(row["frame"]))
                pad = ''
                for i in range(8 - l):
                    pad += '0'
                frame = str(row["frame"])

                # Fetch image
                file_path = data_paths[j] + image_path + pad + frame + '.png'
                img = cv2.imread(file_path)
                if len(img) == 0:
                    print("Error fetching image")

                # Converting to rgb
                img = img[..., ::-1]

                # Cropping
                img = img[self.top_crop:, :,:]
                #img = cv2.resize(img,(200,88))

                # Add to recording
                measurment_recording.at[index, "Image"] = np.array(img)

        self.data = pd.concat(self.data_as_recordings, ignore_index=True)

    def set_train_valid_split(self, test_size=0.25): 
        data = self.get_data(as_one_list=True)
        split = train_test_split(data, test_size=0.25)
        (self.training_data, self.validation_data) = split

    
    def set_XY_data(self, atrX, atrY, train_valid_split=True):
        if train_valid_split == True:        
            self.trainX, self.validX = self.get_attributes(atrX)
            self.trainY, self.validY = self.get_attributes(atrY)
        else: 
            self.dataX = self.get_attributes(atrX, train_valid_split=False)
            self.dataY = self.get_attributes(atrY, train_valid_split=False)
    
    def get_attributes(self, atr, train_valid_split=True):
        if train_valid_split == True:        
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

    def get_one_hot_encoded(self, categories, values):
        ### FIT to categories
        label_encoder = LabelEncoder()
        label_encoder.fit(categories)
        integer_encoded = label_encoder.transform(categories)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded = onehot_encoder.fit(integer_encoded)
        ### Encode values
        integer_encoded = label_encoder.transform(values)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.transform(integer_encoded)
        return onehot_encoded


    def get_values_as_numpy_arrays(self, values):
        new_shape = values.shape + values[0].shape
        new_values = []
        for value in values:
            new_values.append(value)
        
        return np.array(new_values).reshape(new_shape)

    def filter_input(self, df):
        #df = df.drop(df[(df.score < 50) & (df.score > 20)].index)
        df = df.drop(df[(np.power(df.Steer,2) < 0.001) & (random.randint(0,10) > 3) ].index)
        return df

    def plot_data(self, data=None, images=None):
        if data == None:
            data = self.get_data(as_one_list=True)
        if images == None:
            images = data.Image.values

        fig = plt.figure(figsize=(8,5))
        for i in range(1, 10):
            r = np.random.randint(len(images))
            img = images[r]
            
            steering = "steering: " + str(round(float(data.loc[r, "Steer"]), 3))
            fig.add_subplot(3,3,i, title=steering)
            imgplot = plt.imshow(img)
        fig.tight_layout()
        plt.show() 
