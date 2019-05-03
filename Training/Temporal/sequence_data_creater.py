""" Used to list all folders"""
import os
import cv2
import pandas as pd
import numpy as np
import sys
sys.path.append("../")
from Misc.data_configuration import Config
from Misc.preprocessing import get_one_hot_encoded
class SequenceCreator(object):
    """
    Creates a file(store.h5) that contains all control signals.
    Converts images to sequences and store these in a sepparate folder.
    Currently the class that is preprocessing the images and the control signals.
    """
    #pylint: disable=superfluous-parens

    def __init__(self):

        self.data = None  # Contains all data
        self.data_as_recordings = []  # Contains all data divided into recordings
        self.conf = Config()

        self.bottom_crop = self.conf.bottom_crop
        self.top_crop = self.conf.top_crop

        self.data_paths = []

        self.fetch_folders()
        self.fetch_measurments()
        self.create_store()
        self.create_and_save_img_sequences()

    def fetch_folders(self):
        """ Fetches paths of available folders """
        for folder in os.listdir('../../Training_data'):
            if folder == ".DS_Store" or folder == "store.h5":
                continue
            self.data_paths.append("../../Training_data/" + folder)
        self.data_paths.sort(key=lambda a: int(a.split("/")[-1]))

    def fetch_measurments(self):
        """Fetches measurements"""
        measurements_path = "/Measurments/recording.csv"
        for path in self.data_paths:
            measurement_recording = pd.read_csv(path + measurements_path)
            self.data_as_recordings.append(measurement_recording)

    def create_store(self):
        """Preprocesses control signals and adds them to store.h5"""
        #pylint: disable=line-too-long

        store = pd.HDFStore("../../Training_data/" + 'store.h5')
        for j, (path, measurement_recording) in enumerate(zip(self.data_paths, self.data_as_recordings)):

            # converts image rows to sequence rows and removes 1/3 of measurement with low steering
            temp = self.convert_to_sequences(measurement_recording, self.conf.input_size_data["Sequence_length"])

            temp = self.preprocess_measurements(temp)

            new_path = "Recording_" + path.split("/")[-1]
            print(new_path + " stored")
            store[new_path] = temp
            self.data_as_recordings[j] = temp
        store.close()

    def create_and_save_img_sequences(self):
        """Stores images as sequences"""
        image_path = "/Images/"
        for j, measurment_recording in enumerate(self.data_as_recordings):
            try:
                os.mkdir(self.data_paths[j] + "/Sequences")
            except OSError as err:
                print("Failed to create folder: " + \
                     self.data_paths[j] + "/Sequences" + ". " + err.strerror)

            for _, row in measurment_recording.iterrows():
                temp_images = []
                frames = row["Frames"]
                cur_frame = row["frame"]

                for f in frames:
                    # Add padding to frame number
                    frame_len = len(str(f))
                    pad = ''
                    for _ in range(8 - frame_len):
                        pad += '0'
                    frame = str(f)

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

    def preprocess_measurements(self, dataframe):
        """ preprocesses data before storing """

        # Insert one_hot_encodings into the data-frame and convert speed to numpy array??
        for index, row in dataframe.iterrows():
            # Convert Directions to one_hot_encoding
            if self.conf.input_data["Direction"]:
                ohe_directions = get_one_hot_encoded(
                    self.conf.direction_categories,
                    row.Direction
                )
                #print(ohe_directions)
                dataframe.at[index, "Direction"] = ohe_directions

            # Convert TL_states to one_hot_encoding
            if self.conf.input_data["TL_state"]:
                ohe_tl_state = get_one_hot_encoded(
                    self.conf.tl_categories,
                    row.TL_state
                )
                dataframe.at[index, "TL_state"] = ohe_tl_state

            #TODO: Verify if necesarry
            if self.conf.input_data["Speed"]:
                speed = np.array([dataframe.loc[index, "Speed"]])
                dataframe.at[index, "Speed"] = speed

        return dataframe


    def convert_to_sequences(self, dataframe, sequence_length):
        """Filters away all features that shouldn't be used and convert"""
        new_df = pd.DataFrame()
        for index, row in dataframe.iterrows():
            if index < sequence_length - 1:
                continue
            else:
                temp_df = dataframe.iloc[index - sequence_length + 1: index + 1, :]

                #TODO: Convert to for loop over all measures
                temp_steerings = temp_df.loc[:, "Steer"].values
                temp_directions = temp_df.loc[:, "Direction"].values
                temp_frames = temp_df.loc[:, "frame"].values.astype(int)
                temp_tl_states = temp_df.loc[:, "TL_state"].values
                # Add frames and directions too the new row
                new_row = row
                new_row.at["Frames"] = temp_frames
                new_row.at["Direction"] = temp_directions
                new_row.at["Steer"] = temp_steerings
                new_row.at["TL_state"] = temp_tl_states

                new_df = new_df.append(new_row, ignore_index=True)

        new_df["frame"] = new_df["frame"].astype(int)
        return new_df


SequenceCreator()
