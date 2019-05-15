import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

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

def filter_input_based_on_steering(dataframe, conf, temporal):
    """ Filters dataframe consisting of sequences based on steering """
    if not temporal:
        new_df = dataframe.drop(
            dataframe[
                (np.abs(dataframe.Steer) < conf.filter_threshold) & \
                (dataframe.Direction == "RoadOption.LANEFOLLOW") & \
                (random.randint(0, 10) > (10 - (10 * conf.filtering_degree)))
            ].index)
        return new_df

    sequence_length = conf.input_size_data["Sequence_length"]
    new_df = pd.DataFrame()
    for _, row in dataframe.iterrows():
        # Add or remove the current sequence
        directions = row["Direction"]
        steerings = row["Steer"]
        follow_lane = True
        if temporal:
            for direction in directions:
                if direction[2] == 0:
                    follow_lane = False
            sum_steering = sum(abs(steerings))

        #pylint: disable=line-too-long
        if temporal:
            if follow_lane and \
                sum_steering/sequence_length < conf.filter_threshold and \
                random.randint(0, 10) > (10 - (10 * conf.filtering_degree)):
                continue
            else:
                new_df = new_df.append(row, ignore_index=True)  
        return new_df

def get_values_as_numpy_arrays(values):
    new_shape = values.shape + values[0].shape
    new_values = []
    for value in values:
        new_values.append(value)
    
    return np.array(new_values).reshape(new_shape)
