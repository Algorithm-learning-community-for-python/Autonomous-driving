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
    print("\n")
    print("\n")
    print("-------------------- FILTERING DATASET BASED ON STEERING -----------------------")
    try:
        _ = dataframe.ohe_speed_limit
        
        speed_limit_rep = ("ohe_speed_limit", "[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]")
    except AttributeError:
        speed_limit_rep = ("speed_limit", 0.3)
    if not temporal:
        droppable = dataframe[(
            (np.abs(dataframe.Steer) < conf.filter_threshold) & \
            ((dataframe.Direction == "[0. 0. 1. 0. 0. 0. 0.]") | (dataframe.Direction == "RoadOption.LANEFOLLOW")) & \
            (dataframe[speed_limit_rep[0]] == speed_limit_rep[1])  # only filtering from 30km/h since we still want a lot of data from high speed
        )].index
        s = int(np.ceil(conf.filtering_degree*len(droppable)))
        print("Dropping " + str(s) + " out of " + str(len(droppable)) + " droppable, and " + str(len(dataframe.index)) + " total" )
        droppable = np.random.choice(droppable, size=s, replace=False)
        new_df = dataframe.drop(droppable)
        print("Dataset size after filtering: " + str(len(new_df.index)))
        print("\n")
        print("\n")
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
        print("\n")
        print("\n")
        return new_df


def filter_input_based_on_speed_and_tl(dataframe, conf, temporal):
    """ Filters dataframe consisting of sequences based on steering """
    print("\n")
    print("\n")
    print("-------------------- FILTERING AWAY SAMPLES WHERE THE CAR IS STANDING STILL DUE TO RED LIGHT -----------------------")
    if not temporal:
        droppable = dataframe[
                ((np.abs(dataframe.Speed) < conf.filter_threshold_speed) & \
                ((dataframe.TL_state == "[0. 1. 0.]") | (dataframe.TL_state == "Red")))
            ].index
        s = int(np.ceil(conf.filtering_degree_speed*len(droppable)))
        print("Dropping " + str(s) + " out of " + str(len(droppable)) + " droppable, and " + str(len(dataframe.index)) + " total" )
        droppable = np.random.choice(droppable, size=s, replace=False)
        new_df = dataframe.drop(droppable)
        print("Dataset size after filtering: " + str(len(new_df.index)))
        print("\n")
        print("\n")
        return new_df
    else:
        print("Temporal filtering on speed and tl not implemented...")
        print("exiting...")
        exit()

def filter_corrupt_input(dataframe, conf, temporal):
    """ Filters dataframe consisting of sequences based on steering """
    print("\n")
    print("\n")
    print("-------------------- FILTERING AWAY CORRUPT DATA -----------------------")
    if not temporal:
        droppable = dataframe[(
                ((dataframe.Steer == 1) & \
                ((dataframe.Direction == "[0. 0. 0. 1. 0. 0. 0.]") | (dataframe.Direction == "RoadOption.LEFT")))
                |
                ((dataframe.Steer == 1) & \
                ((dataframe.Direction == "[0. 0. 0. 0. 0. 1. 0.]") | (dataframe.Direction == "RoadOption.STRAIGHT")))
        )].index
        #s = int(np.ceil(conf.filtering_degree_speed*len(droppable)))
        print("Dropping " + str(len(droppable)) + " out of " + str(len(dataframe.index)) + " total" )
        #droppable = np.random.choice(droppable, size=s, replace=False)
        new_df = dataframe.drop(droppable)
        print("Dataset size after filtering: " + str(len(new_df.index)))
        print("\n")
        print("\n")
        return new_df
    else:
        print("Temporal filtering on speed and tl not implemented...")
        print("exiting...")
        exit()

def get_values_as_numpy_arrays(values):
    new_shape = values.shape + values[0].shape
    new_values = []
    for value in values:
        new_values.append(value)
    
    return np.array(new_values).reshape(new_shape)
