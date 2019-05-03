# KERAS IMPORTS
from keras.optimizers import Adam
from keras.utils import plot_model

from keras.callbacks import ModelCheckpoint

# SCIKIT-Learn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Div
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle


#local Classes
from data_handler_spatial import DataHandler
from network import NetworkHandler, load_network

class Conf:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class Trainer():
    def __init__(self):
        self.atrX=["Direction", "Image"]# "speed_limit" , "Speed",  "TL_state",
        self.atrY=["Steer"] #, "Throttle", "Brake"

        self.input_data = {
            "Direction": True,
            "Speed": False,
            "SL": False,
            "TL": False,
        }
        self.input_size_data ={
            "Image": [140, 320,3],
            "Direction": [7],
            "Speed": [1],
            "SL": [1],
            "TL": [3],
            "Output": 1,
        }
        self.direction_categories = [
            "RoadOption.VOID", 
            "RoadOption.LEFT",
            "RoadOption.RIGHT",
            "RoadOption.STRAIGHT",
            "RoadOption.LANEFOLLOW",
            "RoadOption.CHANGELANELEFT",
            "RoadOption.CHANGELANERIGHT"
        ]

        self.tl_categories = [
            "Green",
            "Yellow",
            "Red"
        ]
        lr = 0.0001
        args = {
            "loss": "mse",
            "optimizer": Adam(lr),
            "lr": lr,
            "metrics": None,
            "epochs": 2,
            "batch_size": 32,
        }
        self.conf = Conf(**args)
        print(self.conf.__dict__)
        self.train_valid_split=0.2
        self.data_handler = None
        self.network_handler = None
        self.history = None
        self.optimizer = Adam()
        self.model = None

        self.init_data_handler()
        self.init_network()

    def init_data_handler(self):
        # Sets Data, dataX and dataY
        self.data_handler = DataHandler(atrX=self.atrX, atrY=self.atrY, train_valid_split=self.train_valid_split)
        
        #Convert fields to one_hot_encoding
        ohe_directions = self.data_handler.get_one_hot_encoded(self.direction_categories, self.data_handler.data.Direction)
        ohe_tl_state = self.data_handler.get_one_hot_encoded(self.tl_categories, self.data_handler.data.TL_state)
        
        # Insert one_hot_encoding into the dataframe
        if self.input_data["Direction"]:
            for index, _ in self.data_handler.data.iterrows():
                self.data_handler.data.at[index, "Direction"] = ohe_directions[index]
        
        if self.input_data["TL"]:
            for index, _ in self.data_handler.data.iterrows():
                self.data_handler.data.at[index, "TL_state"] = ohe_tl_state[index]
        if self.input_data["Speed"]:
            for index, _ in self.data_handler.data.iterrows():
                speed = np.array([self.data_handler.data.loc[index,"Speed"]])
                self.data_handler.data.at[index, "Speed"] = speed
        
        # Set dataX, dataY
        self.data_handler.set_XY_data(self.atrX, self.atrY, train_valid_split=False)
        # Set training_data, validation_data,
        self.data_handler.set_train_valid_split(self.train_valid_split)
        # TrainX,TrainY, ValidX,ValidY
        self.data_handler.set_XY_data(self.atrX, self.atrY, train_valid_split=True)

    def init_network(self):
        self.network_handler = load_network(self.input_size_data, self.input_data)
        plot_model(self.network_handler.model, to_file='model.png')


    def train(self):
        self.model = self.network_handler.model
        
        train_dir = self.data_handler.trainX.Direction.values
        train_dir = self.data_handler.get_values_as_numpy_arrays(train_dir)

        valid_dir = self.data_handler.validX.Direction.values
        valid_dir = self.data_handler.get_values_as_numpy_arrays(valid_dir)

        train_img = self.data_handler.trainX.Image.values
        train_img = self.data_handler.get_values_as_numpy_arrays(train_img)

        valid_img = self.data_handler.validX.Image.values
        valid_img = self.data_handler.get_values_as_numpy_arrays(valid_img)

        trainY = self.data_handler.trainY.values
        validY = self.data_handler.validY.values
        
        self.model.compile(loss=self.conf.loss, optimizer=self.conf.optimizer, metrics=self.conf.metrics)
        self.history = self.model.fit(
            [train_img, train_dir],
            trainY,
            validation_data=([valid_img, valid_dir], validY),
            epochs=self.conf.epochs,
            batch_size=self.conf.batch_size
        )
        self.model.save('model.h5')

    def store_results(self, folder=None):
        if folder:
            folder = folder
        else:
            last_folder = 0
            for folder in os.listdir('Stored_models'):
                if int(folder) >= last_folder:
                    last_folder = int(folder)+1
            folder = last_folder 

        path = "Stored_models/" + str(folder)

        try:  
            os.mkdir(path)

        except OSError:  
            print ("Creation of the directory %s failed" % path)
        else:  
            print ("Successfully created the directory %s " % path)
        
        # Stores to mmuch
        #with open(path +"/Trainer" + ".pkl", "wb+") as f:
        #    pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        #Store model
        self.model.save(path + "/model.h5")

        #Store image of model
        plot_model(self.network_handler.model, to_file=path + '/model.png')


        f = open(path + "/conf.txt", "wb+")
        # Store config
        f.write(str(self.conf.__dict__)+ "\n")
        # Store settings
        f.write(str(self.input_data)+ "\n")
        f.write(str(self.input_size_data)+ "\n")
        #Store training history
        f.write(str(self.history.history)+ "\n")
        f.close()

    def plot_training_results(self):
        # summarize history for accuracy
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

trainer = Trainer()
#trainer.data_handler.plot_data()
trainer.train()
trainer.store_results()
#trainer.plot_training_results()