"""Temporal generator"""
from math import ceil
import random
import pandas as pd
import numpy as np
from keras.utils import Sequence
from Misc.misc import get_image, get_data_paths
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

class BatchGenerator(Sequence):
    """ Generator that yields batches of training data concisting of multiple inputs"""
    #pylint: disable=too-many-instance-attributes
    def __init__(self, conf, data="Training_data"):
        self.conf = conf
        self.batch_size = self.conf.train_conf.batch_size
        self.seq_len = self.conf.input_size_data["Sequence_length"]
        self.data = None
        self.data_paths = get_data_paths(data)
        self.input_measures = [
            key for key in self.conf.available_columns if self.conf.input_data[key]
            ]
        self.output_measures = [
            key for key in self.conf.available_columns if self.conf.output_data[key]
            ]
        self.get_measurements_recordings(data)

    def __len__(self):
        return np.ceil(len(self.data)/self.batch_size)

    def __getitem__(self, idx):
        images = []
        measurments =  []
        for b in range(self.batch_size):
            images.append([])

        for i in range(len(self.input_measures)):
            measurments.append([])
        y = []
        for i in range(len(self.output_measures)):
            y.append([])

        cur_idx = idx*self.batch_size
        for b in range(self.batch_size):
            #print("new_batch")
            # Add the current sequence to the batch
            sequence = self.data[cur_idx]
            for j in range(self.seq_len):
                #print("new_row")
                row = sequence.iloc[j, :]
                #print(row.frame)
                images[b].append(self.get_image(row))
            
            row = sequence.iloc[-1, :]
            input_measurements = row[self.input_measures]
            output_measurements = row[self.output_measures]
            for i, measure in enumerate(self.input_measures):
                #if measure == "Speed":
                #    measurments[i].append([input_measurements[measure]])
                #else:
                measurments[i].append(input_measurements[measure])

            for i, measure in enumerate(self.output_measures):
                y[i].append(output_measurements[measure])
            cur_idx += 1

        # Convert x to dict to allow for multiple inputs
        X = {}
        X["input_Image"] = np.array(images)
        for i, measure in enumerate(self.input_measures):
            X["input_" + measure] = np.array(measurments[i])
        Y = {}
        for i, measure in enumerate(self.output_measures):
            Y["output_" + measure] = np.array(y[i])
        return X, Y

    def get_measurements_recordings(self, data):
        self.data = []
        for i, path in enumerate(self.data_paths): #[:int(len(self.data_paths)/10)]):
            df = pd.read_csv(path + self.conf.recordings_path)
            df["Recording"] = i 
            for i in range(len(df)):
                if i + self.seq_len < len(df):
                    self.data.append(df.iloc[i:i + self.seq_len, :].copy())

        #Filter
        if self.conf.filter_input and data != "Validation_data":
            self.data = filter_input_based_on_steering(self.data, self.conf, temporal=True)
            self.data = filter_input_based_on_speed_and_tl(self.data, self.conf, temporal=True)
            self.data = filter_corrupt_input(self.data, self.conf, temporal=True)
        for i, sequence in enumerate(self.data):
            for index, row in sequence.iterrows():
                if self.conf.input_data["Direction"]:
                    self.data[i].at[index, "Direction"] = [int(x) for x in str(row["Direction"]).strip("][").split(".")[:-1]]
                if self.conf.input_data["TL_state"]:
                    self.data[i].at[index, "TL_state"] = [int(x) for x in str(row["TL_state"]).strip("][").split(".")[:-1]]
                if self.conf.input_data["ohe_speed_limit"]:
                    self.data[i].at[index, "ohe_speed_limit"] = [int(x) for x in str(row["ohe_speed_limit"]).strip("][").split(".")[:-1]]

    def get_image(self, row):
        """" Returns the image sequence"""
        path = self.data_paths[row["Recording"]] + "/Images/"
        frame = str(row["frame"])
        img = get_image(path, frame)

        img = img[..., ::-1]
        img = img[self.conf.top_crop:, :, :]
        return img


def get_one_hot_encoded(categories, values):
    """ Returns one hot encoding of categories based on values"""
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

def filter_input_based_on_steering(sequences, conf, temporal):
    """ Filters dataframe consisting of sequences based on steering """
    print("\n")
    print("\n")
    print("-------------------- FILTERING DATASET BASED ON STEERING -----------------------")
    try:
        _ = sequences[0].ohe_speed_limit
        
        speed_limit_rep = ("ohe_speed_limit", "[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]")
    except AttributeError:
        speed_limit_rep = ("speed_limit", 0.3)

    sequence_length = conf.input_size_data["Sequence_length"]
    count = 0
    l = len(sequences)
    for i, sequence in enumerate(sequences):
        # Add or remove the current sequence
        directions = sequence["Direction"]
        steerings = sequence["Steer"]
        sum_steering = sum(abs(steerings))
        speed_limits = sequence[speed_limit_rep[0]]
        follow_lane = True
        speed_limit_30 = True
        low_steering = True
        for direction in directions:
            if direction != "[0. 0. 1. 0. 0. 0. 0.]":
                follow_lane = False
        for speed_limit in speed_limits:
            if speed_limit != speed_limit_rep[1]:
                speed_limit_30 = False
        for steering in steerings:
            if steering > conf.filter_threshold:
                low_steering = False
        
        if sum_steering > conf.filter_threshold*sequence_length:
            low_steering = False
        
        drop = random.randint(0, 10) > (10 - (10 * conf.filtering_degree))
        
        if follow_lane and speed_limit_30 and low_steering and drop:
            sequences.pop(i)
            count += 1

    print("Dropped " + str(count) + " out of " + str(l))
    print("Dataset size after filtering: " + str(len(sequences)))
    print("\n")
    print("\n")
    return sequences


def filter_input_based_on_speed_and_tl(sequences, conf, temporal):
    """ Filters dataframe consisting of sequences based on steering """
    print("\n")
    print("\n")
    print("-------------------- FILTERING AWAY SAMPLES WHERE THE CAR IS STANDING STILL DUE TO RED LIGHT -----------------------")
    count = 0
    l = len(sequences)
    for i, sequence in enumerate(sequences):
        # Add or remove the current sequence
        Speeds = sequence["Speed"]
        tl_states = sequence["TL_state"]
        speed_is_low = True
        tl_is_red = True
        for speed in Speeds:
            if speed > conf.filter_threshold_speed:
                speed_is_low = False
        for tl_state in tl_states:
            if tl_state != "[0. 1. 0.]":
                tl_is_red = False

        random_choice = random.randint(0, 10) > (10 - (10 * conf.filtering_degree_speed))
        
        if speed_is_low and tl_is_red and random_choice:
            sequences.pop(i)
            count += 1

    print("Dropped " + str(count) + " out of " + str(l))
    print("Dataset size after filtering: " + str(len(sequences)))
    print("\n")
    print("\n")
    return sequences

def filter_corrupt_input(sequences, conf, temporal):
    """ Filters dataframe consisting of sequences based on steering """
    print("\n")
    print("\n")
    print("-------------------- FILTERING AWAY CORRUPT DATA -----------------------")
    count = 0
    l = len(sequences)
    for i, sequence in enumerate(sequences):
        steers = sequence["Steer"]
        directions = sequence["Direction"]
   
        if (steers.values[-1] == 1 and directions.values[-1] == "[0. 0. 0. 1. 0. 0. 0.]") or (steers.values[-1] == 1 and directions.values[-1] == "[0. 0. 0. 0. 0. 1. 0.]"):
            print("dropped because error: \n" + str(sequence))
            count += 1
            sequences.pop(i)
    print("Dropped " + str(count) + " out of " + str(l))
    print("Dataset size after filtering: " + str(len(sequences)))
    print("\n")
    print("\n")
    return sequences