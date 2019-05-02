# Div
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
import numpy as np

class DataHandler():
    def __init__(self):
        self.measurements = []
        self.images = []

        self.training_images = None
        self.testing_images = None

        self.training_data_X = None
        self.test_data_X = None

        self.trainY = None
        self.testY = None


    def fetch_data(self):
        measurements_path = "/Measurments/recording.csv"
        image_path = "/Images/"

        data_paths=[]
        for folder in os.listdir('../Training_data'):
            data_paths.append("../Training_data/"+folder)
        
        self.fetch_measurements(data_paths, measurements_path)
        self.fetch_images(data_paths, image_path)

    def fetch_measurements(self, data_paths, measurements_path):
        for path in data_paths:
            measurment_recording = pd.read_csv(path + measurements_path )
            self.measurements.append(measurment_recording)

    def fetch_images(self, data_paths, image_path):
        self.images = []
        self.frames = []
        for j, measurment_recording in enumerate(self.measurements):
            image_recording = []
            frame_recording = []
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
                # Add to recording
                image_recording.append(img)
                frame_recording.append(frame)
            # Add to full collection
            self.images.append(image_recording)
            self.frames.append(frame_recording)
    
    def get_measurements(self, as_one_list=True):
        if as_one_list:
            return(pd.concat(self.measurements, ignore_index=True))
        else:
            return self.measurements
    
    def get_images(self, as_one_list=True):
        if as_one_list:
            images = []
            for i in range(len(self.images)):
                images.extend(self.images[i])
            return images
        else:
            return self.images

    def get_frames(self, as_one_list=True):
        if as_one_list:
            frames = []
            for i in range(len(self.frames)):
                frames.extend(self.frames[i])
            return frames
        else:
            return self.frames
    
    def set_train_test_split(self, measurements=None, images=None, Xatr=[], Yatr=["Steer"], test_size=0.25): 
        if measurements == None:
            measurements = self.get_measurements(as_one_list=True)
        if images == None:
            images = self.get_images(as_one_list=True)

        split = train_test_split(measurements, images, test_size=0.25)
        (trainAttrX, testAttrX, trainImgX, testImgX) = split

        #Set images
        self.training_images = trainImgX
        self.testing_images = testImgX

        #Set X attributes
        if len(Xatr) > 0:
            self.training_data_X = trainAttrX.loc[:,Xatr]
            self.test_data_X = testAttrX.loc[:,Xatr]

        else: 
            self.training_data_X = None

        #Set y Attriibutes
        self.trainY = trainAttrX.loc[:,Yatr]
        self.testY = testAttrX.loc[:,Yatr]


    def get_training_data(self, as_numpy=True):
        if as_numpy:
            return np.array(self.training_images), self.training_data_X.values, self.trainY.values
        else:
            return self.training_images, self.training_data_X, self.trainY
    

    def get_test_data(self, as_numpy=True):
        if as_numpy:
            return np.array(self.testing_images), self.test_data_X.values, self.testY.values
        else:
            return self.testing_images, self.test_data_X, self.testY
    

    def plot_data(self, measurements=None, images=None):
        if measurements == None:
            measurements = self.get_measurements(as_one_list=True)
        if images == None:
            images = self.get_images(as_one_list=True)
        
        fig = plt.figure(figsize=(8,5))
        for i in range(1, 10):
            r = np.random.randint(len(images))
            img = images[r]
            
            steering = "steering: " + str(round(float(measurements.loc[r, "Steer"]), 3))
            fig.add_subplot(3,3,i, title=steering)
            imgplot = plt.imshow(img)
        fig.tight_layout()
        plt.show() 

    def test_image_data_correspondance(self):
        measurements = self.get_measurements(as_one_list=True)
        frames = self.get_frames(as_one_list=True)
        for i, frame in enumerate(frames):
            if(frame != str(measurements.loc[i, "frame"])):
                warnings.warn("image frame " + str(frame) + " doesn't correspond with data frame " + str(measurements.loc[i, "frame"]))
        print("test_image_data_correspondance complited")