# Div
import cv2
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
from data_configuration import Config

# TODO: REMOVE COLLUMNS NOT IN USE


class SequenceCreator:
    def __init__(self):

        self.data = None  # Contains all data
        self.data_as_recordings = []  # Contains all data divided into recordings
        self.conf = Config()
        self.bottom_crop = 115
        self.top_crop = 70

        self.data_paths = None

        self.fetch_data()

    def fetch_data(self):
        # Fetch measurements
        self.fetch_measurments()

        # Fix measurement data into correct format
        # Then store to the dataframe
        store = pd.HDFStore("../Training_data/" + 'store.h5')
        for j, measurement_recording in enumerate(self.data_as_recordings):

            # add field Frames and remove 1/3 of measurement with low steering
            temp = self.convert_and_filter_input(measurement_recording, sequence_length=5)

            # Convert fields to one_hot_encoding
            ohe_directions = self.get_one_hot_encoded(self.conf.direction_categories, measurement_recording.Direction)
            ohe_tl_state = self.get_one_hot_encoded(self.conf.tl_categories, measurement_recording.TL_state)

            # Insert one_hot_encoding into the data-frame
            if self.conf.input_data["Direction"]:
                for index, _ in temp.iterrows():
                    temp.at[index, "Direction"] = ohe_directions[index]

            if self.conf.input_data["TL_state"]:
                for index, _ in temp.iterrows():
                    temp.at[index, "TL_state"] = ohe_tl_state[index]

            if self.conf.input_data["Speed"]:
                for index, _ in temp.iterrows():
                    speed = np.array([temp.loc[index, "Speed"]])
                    temp.at[index, "Speed"] = speed

            store[str(j)] = temp
            self.data_as_recordings[j] = temp

        store.close()
        # Store images as sequences
        self.create_and_store_img_sequences()

    def fetch_measurments(self):
        measurements_path = "/Measurments/recording.csv"
        data_paths = []

        for folder in os.listdir('../Training_data'):
            if folder == ".DS_Store" or folder == "store.h5":
                continue
            data_paths.append("../Training_data/" + folder)
        self.data_paths = data_paths
        for path in data_paths:
            measurement_recording = pd.read_csv(path + measurements_path)
            self.data_as_recordings.append(measurement_recording)

    def create_and_store_img_sequences(self):
        image_path = "/Images/"
        for j, measurment_recording in enumerate(self.data_as_recordings):
            try:
                os.mkdir(self.data_paths[j] + "/Sequences")
            except OSError as err:
                print("Failed to create path, " + err.strerror)

            for index, row in measurment_recording.iterrows():
                temp_images = []
                frames = row["Frames"]
                cur_frame = row["frame"]

                for fr in frames:
                    # Add padding to frame number
                    frame_len = len(str(fr))
                    pad = ''
                    for i in range(8 - frame_len):
                        pad += '0'
                    frame = str(fr)

                    # Fetch image
                    file_path = self.data_paths[j] + image_path + pad + frame + '.png'
                    img = cv2.imread(file_path)

                    if len(img) == 0:
                        print("Error fetching image")

                    # Converting to rgb
                    img = img[..., ::-1]

                    # Cropping
                    img = img[self.top_crop:, :, :]

                    temp_images.append(img)
                np.save(self.data_paths[j] + "/Sequences/" + str(cur_frame), np.array(temp_images))

    @staticmethod
    def get_one_hot_encoded(categories, values):
        # FIT to categories
        label_encoder = LabelEncoder()
        label_encoder.fit(categories)
        integer_encoded = label_encoder.transform(categories)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoder = onehot_encoder.fit(integer_encoded)
        # Encode values
        integer_encoded = label_encoder.transform(values)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.transform(integer_encoded)
        return onehot_encoded

    @staticmethod
    def convert_and_filter_input(df, sequence_length=5):
        new_df = pd.DataFrame()
        for index, row in df.iterrows():
            if index < sequence_length - 1:
                continue
            else:
                temp_df = df.iloc[index - sequence_length + 1: index + 1, :]

                temp_steer = temp_df.loc[:, "Steer"].values
                temp_dir = temp_df.loc[:, "Direction"].values
                follow_lane = True
                for direction in temp_dir:
                    if direction != "RoadOption.LANEFOLLOW":
                        follow_lane = False

                # Add or remove the current sequence
                s = sum(np.power(temp_steer, 2))
                if follow_lane and s < (float(sequence_length) / 500) and random.randint(0, 10) > 3:
                    continue
                else:

                    new_row = row
                    new_row.at["Frames"] = temp_df.loc[:, "frame"].values.astype(int)
                    new_df = new_df.append(new_row, ignore_index=True)

        new_df["frame"] = new_df["frame"].astype(int)
        return new_df


SequenceCreator()
