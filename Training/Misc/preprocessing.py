""" Various method used during preprocessing of training data """
#pylint: disable=superfluous-parens
#pylint: disable=line-too-long
#pylint: disable=invalid-name


import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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

def filter_input_based_on_steering(dataframe, conf):
    """ Filters dataframe consisting of sequences based on steering """
    print("\n")
    print("-------------------- FILTERING DATASET BASED ON STEERING -----------------------")
    try:
        _ = dataframe.ohe_speed_limit
        speed_limit_rep_20 = ("ohe_speed_limit", "[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]")
        speed_limit_rep_30 = ("ohe_speed_limit", "[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]")

    except AttributeError:
        speed_limit_rep_20 = ("speed_limit", 0.2)
        speed_limit_rep_30 = ("speed_limit", 0.3)

    droppable = dataframe[(
        (np.abs(dataframe.Steer) < conf.filter_threshold) & \
        ((dataframe.Direction == "[0. 0. 1. 0. 0. 0. 0.]") | (dataframe.Direction == "[0. 0. 0. 0. 0. 0. 1.]") | (dataframe.Direction == "RoadOption.LANEFOLLOW")) & \
        ((dataframe[speed_limit_rep_20[0]] == speed_limit_rep_20[1]) | (dataframe[speed_limit_rep_30[0]] == speed_limit_rep_30[1]))  # only filtering from 20km/h and 30km/h since we still want a lot of data from high speed
    )].index
    count_dropped = int(np.ceil(conf.filtering_degree*len(droppable)))
    print("Dropping " + str(count_dropped) + " out of " + str(len(droppable)) + " droppable, and " + str(len(dataframe.index)) + " total")
    droppable = np.random.choice(droppable, size=count_dropped, replace=False)
    new_df = dataframe.drop(droppable)
    print("Dataset size after filtering: " + str(len(new_df.index)))
    print("\n")
    return new_df

def filter_sequence_input_based_on_steering(sequences, conf):
    """ Filters dataframe consisting of sequences based on steering """
    print("\n")
    print("-------------------- FILTERING DATASET BASED ON STEERING -----------------------")
    try:
        _ = sequences[0].ohe_speed_limit
        speed_limit_rep_20 = ("ohe_speed_limit", "[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]")
        speed_limit_rep_30 = ("ohe_speed_limit", "[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]")
    except AttributeError:
        speed_limit_rep_20 = ("speed_limit", 0.2)
        speed_limit_rep_30 = ("speed_limit", 0.3)

    count_dropped = 0
    filtered_sequences = []
    for sequence in sequences:
        # Add or remove the current sequence
        follow_lane = True
        speed_limit_20 = True
        low_steering = True
        no_car_in_front = True
        # check if any of the images is from a turn
        for direction in sequence["Direction"].values:
            if direction != "[0. 0. 1. 0. 0. 0. 0.]" and direction != "[0. 0. 0. 0. 0. 0. 1.]":
                follow_lane = False

        if not follow_lane:
            filtered_sequences.append(sequence)
            continue

        # Only filter from speed limit 20 (need samples with high speed limits)
        for speed_limit in sequence[speed_limit_rep_30[0]].values:
            if speed_limit != speed_limit_rep_30[1]:
                speed_limit_20 = False
        if not speed_limit_20:
            filtered_sequences.append(sequence)
            continue
        # filter away samples where all steerings are below the threshold
        for steering in sequence["Steer"].values:
            if steering > conf.filter_threshold:
                low_steering = False
        if not low_steering:
            filtered_sequences.append(sequence)
            continue

        # Only filter if we can assume that there is no car in front
        for speed in sequence["Speed"].values:
            if speed < 0.1:
                no_car_in_front = False

        for brake in sequence["Brake"].values:
            if brake != 0:
                no_car_in_front = False

        # Only filter away a random subsample
        if random.randint(0, 10) > (10 - (10 * conf.filtering_degree)):
            # filter away the sample
            if no_car_in_front:
                count_dropped += 1
            # Only filter away 40% of the samples where we assume there is a car in front
            elif random.randint(0, 10) > 6:
                count_dropped += 1
            else:
                filtered_sequences.append(sequence)
        else:
            filtered_sequences.append(sequence)

    print("Dropped " + str(count_dropped) + " out of " + str(len(sequences)))
    print("Dataset size after filtering: " + str(len(filtered_sequences)))
    print("\n")
    return filtered_sequences

def filter_input_based_on_speed_and_tl(dataframe, conf):
    """ Filters dataframe consisting of sequences based on steering """
    print("\n")
    print("-------------------- FILTERING AWAY SAMPLES WHERE THE CAR IS STANDING STILL DUE TO RED LIGHT -----------------------")
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
    return new_df

def filter_sequence_input_based_on_speed_and_tl(sequences, conf):
    """ 
    Use filter_sequence_input_based_on_not_moving instead!
    Filters dataframe consisting of sequences based on steering
    """
    print("\n")
    print("-------------------- FILTERING AWAY SAMPLES WHERE THE CAR IS STANDING STILL DUE TO RED LIGHT -----------------------")
    count_dropped = 0
    filtered_sequences = []
    for sequence in sequences:
        # Add or remove the current sequence
        if sequence["Speed"].values[-1] > conf.filter_threshold_speed:
            filtered_sequences.append(sequence)
            continue

        if sequence["TL_state"].values[-1] != "[0. 1. 0.]":
            filtered_sequences.append(sequence)
            continue

        if random.randint(0, 10) > (10 - (10 * conf.filtering_degree_speed)):
            count_dropped += 1            
        else:
            filtered_sequences.append(sequence)

    print("Dropped " + str(count_dropped) + " out of " + str(len(sequences)))
    print("Dataset size after filtering: " + str(len(filtered_sequences)))
    print("\n")
    return filtered_sequences

def filter_sequence_input_based_on_not_moving(sequences, conf):
    """ Filters dataframe consisting of sequences based on the car not moving """
    print("\n")
    print("-------------------- FILTERING AWAY SAMPLES WHERE THE CAR IS STANDING STILL DUE TO BRAKING -----------------------")
    count_dropped = 0
    filtered_sequences = []
    standing_still_count = 0
    full_brake_count = 0
    for sequence in sequences:
        # Add or remove the current sequence
        speeds = sequence["Speed"].values
        brakes = sequence["Brake"].values
        standing_still = True
        braking = True
        #for speed in speeds:
        #    if speed > conf.filter_threshold_speed:
        if speeds[-1] > conf.filter_threshold_speed:
            standing_still = False
        if brakes[-1] != 1:
            braking = False
        #for tl_state in tl_states:
        #    if tl_state != "[0. 1. 0.]":
        #if tl_states[-1] != "[0. 1. 0.]":
        #    tl_is_red = False

        if standing_still:
            standing_still_count += 1
        if braking:
            full_brake_count += 1
        
        drop = random.randint(0, 10) > (10 - (10 * conf.filtering_degree_speed))
        if standing_still and braking and drop:
            count_dropped += 1
        else:
            filtered_sequences.append(sequence)

    print("Dropped " + str(count_dropped) + " out of " + str(len(sequences)))
    #print("Standing still count: " + str(standing_still_count))
    #print("Full brake count: " + str(full_brake_count))
    print("Dataset size after filtering: " + str(len(filtered_sequences)))
    print("\n")
    return filtered_sequences

def filter_corrupt_input(dataframe):
    """ Filters dataframe consisting of sequences based on steering """
    print("\n")
    print("-------------------- FILTERING AWAY CORRUPT DATA -----------------------")
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
    return new_df

def filter_corrupt_sequence_input(sequences):
    """ 
    Filters dataframe consisting of sequences based on steering 
    Should be used with care, since the car might correct the steering during a turn
    """
    print("\n")
    print("-------------------- FILTERING AWAY CORRUPT DATA -----------------------")
    count_dropped = 0
    filtered_sequences = []
    for sequence in sequences:
        steers = sequence["Steer"]
        directions = sequence["Direction"]
        if (steers.values[-1] == 1 and directions.values[-1] == "[0. 0. 0. 1. 0. 0. 0.]") or (steers.values[-1] == 1 and directions.values[-1] == "[0. 0. 0. 0. 0. 1. 0.]"):
            print("dropped because error: \n" + str(sequence))
            count_dropped += 1
        else:
            filtered_sequences.append(sequence)
    print("Dropped " + str(count_dropped) + " out of " + str(len(sequences)))
    print("Dataset size after filtering: " + str(len(filtered_sequences)))
    print("\n")
    return filtered_sequences

def get_values_as_numpy_arrays(values):
    """ Returns the input array converted to an numpy array """
    new_shape = values.shape + values[0].shape
    new_values = []
    for value in values:
        new_values.append(value)
    return np.array(new_values).reshape(new_shape)


def augment_image(image):
    noise_types = ["gauss", "s&p", "poisson", "speckle", "none", "none", "none", "none", ]
    choice = np.random.choice(noise_types)
    if choice =="none":
        return image
    else:
        return noisy(choice, image)

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy