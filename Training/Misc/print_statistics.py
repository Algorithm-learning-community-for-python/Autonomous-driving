import pandas as pd
from preprocessing import (
    filter_input_based_on_steering, \
    filter_input_based_on_speed_and_tl, \
    filter_corrupt_input, \
    filter_input_based_on_not_moving
    )
from Spatial.data_configuration import Config
import os
import matplotlib
matplotlib.use("agg")

from misc import get_data_paths
import matplotlib.pyplot as plt
pd.set_option("display.max_rows", 4000)

def print_column_based_statistics(df, col, name="", count=True, mean=True, std=False):
    total_samples = len(df.index)
    df_grouped = df.groupby(col)
    stats.write("############################ "+ col +" based statistics " + " ############################")
    stats.write("\n")
    stats.write("\n")
    if count:
        stats.write("Sample count " + name)
        stats.write("\n")
        samples_per_direction = df_grouped.count()

        stats.write("Total samples: " + str(total_samples))
        stats.write("\n")
        stats.write("Per " + col + ": ")
        stats.write("\n")

        stats.write(str(samples_per_direction["frame"]))
        stats.write("\n")
        stats.write("\n")

    if mean:
        stats.write("Mean values per " + col)
        stats.write("\n")
        mean_per_direction = df_grouped.mean()

        stats.write("For all samples: ")
        stats.write("\n")

        stats.write(str(df.mean().to_frame().T))
        stats.write("\n")
        stats.write("\n")
        stats.write("Per " + col + ": ")
        stats.write("\n")

        stats.write(str(mean_per_direction))
        stats.write("\n")
        stats.write("\n")

    if std:
        stats.write("Standard deviation " + name)
        stats.write("\n")
        std_per_direction = df_grouped.std()
        stats.write("For all samples: ")
        stats.write("\n")

        stats.write(str(df.std().to_frame().T))
        stats.write("\n")
        stats.write("\n")

        stats.write("Per " + col + ": ")
        stats.write("\n")

        stats.write(str(std_per_direction))
        stats.write("\n")
        stats.write("\n")
    stats.write("\n")
    stats.write("\n")

def print_multi_column_based_statistics(df, columns, target, name, count=True, mean=True, std=False):
    df_grouped = df.groupby(columns)
    seperator = ' and '
    groups = seperator.join(columns)
    stats.write("############################ " + target + " statistics ############################")
    stats.write("\n")

    stats.write("based on " + groups)
    stats.write("\n")
    stats.write("\n")
    
    if count:
        samples_per_direction = df_grouped.count()
        #stats.write("Sample count")
        #stats.write("\n")

        stats.write("Total samples per " + groups + ": ")
        stats.write("\n")

        stats.write(str(samples_per_direction["frame"]))
        stats.write("\n")
        stats.write("\n")

    if mean: 
        mean_per_direction = df_grouped.mean()
        #stats.write("############## Mean values " + name + " ##############")
        #stats.write("\n")
        stats.write("Mean value per " + groups + ": ")
        stats.write("All samples: " + str(df.mean()[target]))
        stats.write("\n")

        stats.write(str(mean_per_direction[target]))
        stats.write("\n")
        stats.write("\n")

    if std:
        std_per_direction = df_grouped.std()
        stats.write("############## Standard deviation " + name + " ##############")
        stats.write("\n")

        stats.write("All samples: " + str(df.std()[target]))
        stats.write("\n")

        stats.write(str(std_per_direction[target]))
        stats.write("\n")
        stats.write("\n")

def print_speed_statistics(df, name=""):
    df["Speed_binned"] = pd.cut(df["Speed"], 10)
    print_column_based_statistics(df, "Speed_binned", name=name)
    stats.write("Number of samples where speed is 0: " + str(df[df['Speed']==0].count()["frame"]))
    stats.write("\n")

    stats.write("Number of samples where speed == 0 and tl is red: " + str(df[(df['Speed']==0) & (df["TL_state"]=="Red")].count()["frame"]))
    stats.write("\n")
    stats.write("\n")
    stats.write("\n")

def print_brake_statistics(df, name=""):
    df["Speed_binned"] = pd.cut(df["Speed"], 10)
    print_multi_column_based_statistics(df, ["TL_state", "Speed_binned"], "Brake", name=name)
    stats.write("\n")
    stats.write("\n")
    stats.write("\n")

def print_throttle_statistics(df, name=""):
    df["Speed_binned"] = pd.cut(df["Speed"], 10)
    print_multi_column_based_statistics(df, ["TL_state", "Speed_binned"], "Throttle", name=name) 
    stats.write("\n")
    stats.write("\n")
    stats.write("\n")

def print_speed_based_on_brake_and_throttle(df, name=""):
    df["Throttle_binned"] = pd.cut(df["Throttle"], 10)
    df["Brake_binned"] = pd.cut(df["Brake"], 10)
    print_multi_column_based_statistics(df, ["Throttle_binned", "Brake_binned"], "Speed", name=name)
    stats.write("\n")
    stats.write("\n")
    stats.write("\n")



def print_steering_statistics(df, conf, name=""):
    df["Steer_binned"] = pd.cut(df["Steer"], 10)
    print_multi_column_based_statistics(df, ["Direction", "Steer_binned"], "frame", name=name, mean=False, std=False)

    stats.write("############## RANGE OF STEERING SAMPLES " + name + " ##############")
    stats.write("\n")

    l = len([(s, d) for (s,d) in zip(df.Steer.values, df.Direction.values) if abs(s) < conf.filter_threshold and d =="RoadOption.LANEFOLLOW"])
    stats.write("Samples steering with absolute value less than " + str(conf.filter_threshold) + " and lanefollow: " + str(l))
    stats.write("\n")

    l = len([x for x in df.Steer.values if abs(x) > conf.filter_threshold])
    stats.write("Samples steering more than " + str(conf.filter_threshold) + ": " + str(l))
    stats.write("\n")
    stats.write("\n")
    stats.write("\n")

def throttle_vs_direction_vs_speed_limit(df, name):
    df["Throttle_binned"] = pd.cut(df["Throttle"], 10)
    df["Steer_binned"] = pd.cut(df["Steer"], 10)
    print_multi_column_based_statistics(df, ["speed_limit", "Direction", "Steer_binned"], "Throttle", name=name)
    stats.write("\n")
    stats.write("\n")
    stats.write("\n")

def print_steering_based_on_direction_and_speed_limit(df, name=""):
    #df["Direction_binned"] = pd.cut(df["Direction"], 10)
    #df["Speed_limit_binned"] = pd.cut(df["speed_limit"], 10)
    print_multi_column_based_statistics(df, ["speed_limit", "Direction"], "Steer", name=name)
    stats.write("\n")
    stats.write("\n")
    stats.write("\n")

all_folders = True
steer_bars = 30
throttle_bars = 20
data_paths=[]
#dataset = "Training_data"
dataset = "Validation_data"
Storage_folder = dataset + "_statistics"
root_paths = os.listdir("../../" + dataset)
all_recordings = []
training_sets = []

conf = Config()
recordings_path = "/Measurments/recording.csv" #conf.recordings_path
if recordings_path == "/Measurments/recording.csv":
    speed_limit_rep = "speed_limit"
else:
    speed_limit_rep = "ohe_speed_limit"


for training_set in root_paths:
    print(training_set)
    training_sets.append(training_set)
    data_paths = get_data_paths(dataset + "/" + training_set)
    recordings = []
    for path in data_paths:
        df = pd.read_csv(path + "/Measurments/recording.csv")
        recordings.append(df)
        all_recordings.append(df)
    if all_folders:
        data = pd.concat(recordings, ignore_index=True)
        try:
            stats = open("../../" + Storage_folder + "/" + training_set + "/spatial_statistics.txt","w") 
        except OSError as ose:
            print(ose)
            print("Creating " + training_set + "folder")
            os.mkdir("../../" + Storage_folder + "/" + training_set)
            stats = open("../../" + Storage_folder + "/" + training_set + "/spatial_statistics.txt","w") 
        
        print("writing statistics:")

        stats.write("\n")
        stats.write("\n")
        stats.write("======================================== DATA STATISTICS ========================================")
        stats.write("\n")
        stats.write("\n")
        stats.write("Training data: " + training_set)
        stats.write("\n")
        stats.write("\n")
        stats.write("---------------------------------------- Before filtering -------------------------------------------")
        stats.write("\n")
        stats.write("\n")

        name = "before filtering"
        print("Direction")
        print_column_based_statistics(data, "Direction", name=name)
        print("TL_state")
        print_column_based_statistics(data, "TL_state", name=name)
        print("ohe_speed_limit")
        print_column_based_statistics(data, speed_limit_rep, name=name)

        #print("brake")
        #print_brake_statistics(data, name)
        #print("throttle")
        #print_throttle_statistics(data, name)
        #print("brake vs throttle")
        #print_speed_based_on_brake_and_throttle(data, name)
        print("print_steering_based_on_direction_and_speed_limit")
        print_steering_based_on_direction_and_speed_limit(data, name)
        print("steering")
        print_steering_statistics(data, conf, name)

        print("plotting")
        fig, ax = plt.subplots()
        ax.hist(data.Steer.values, steer_bars)
        fig.savefig("../../" + Storage_folder + "/" + training_set + "/" + "steer_before_spatial_filtering")

        fig, ax = plt.subplots()
        ax.hist(data.Throttle.values, throttle_bars)
        fig.savefig("../../" + Storage_folder + "/" + training_set + "/" + "throttle_before_spatial_filtering")

        if dataset != "Validation_data":
            data = filter_input_based_on_steering(data, conf)
            data = filter_input_based_on_not_moving(data, conf)
            #data = filter_corrupt_input(data)
            
            print("writing statistics")
            stats.write("\n")
            stats.write("\n")
            stats.write("\n")

            stats.write("---------------------------------------- After filtering -------------------------------------------")
            stats.write("\n")
            stats.write("\n")
            stats.write("\n")
            stats.write("Filtered with: ")
            stats.write("Threshold steering: " + str(conf.filter_threshold))
            stats.write("Degree steering: " + str(conf.filtering_degree))
            stats.write("Threshold speed: " + str(conf.filter_threshold_speed))
            stats.write("Degree speed(standing_still): " + str(conf.filtering_degree_speed))
            stats.write("\n")
            stats.write("\n")

            name = "after filtering"
            print("Direction")
            print_column_based_statistics(data, "Direction", name=name)
            print("TL_state")
            print_column_based_statistics(data, "TL_state", name=name)
            print("ohe_speed_limit")
            print_column_based_statistics(data, speed_limit_rep, name=name)

            #print_brake_statistics(data, name)
            #print_throttle_statistics(data, name)
            #print_speed_based_on_brake_and_throttle(data, name)
            print("print_steering_based_on_direction_and_speed_limit")
            print_steering_based_on_direction_and_speed_limit(data, name)
            print("steering")
            print_steering_statistics(data, conf, name)
            print("plotting")
            fig, ax = plt.subplots()
            ax.hist(data.Steer.values, steer_bars)
            fig.savefig("../../" + Storage_folder + "/" + training_set + "/" + "steer_after_spatial_filtering")

            fig, ax = plt.subplots()
            ax.hist(data.Throttle.values, throttle_bars)
            fig.savefig("../../" + Storage_folder + "/" + training_set + "/" + "throttle_after_spatial_filtering")
        else:
            print("Filtering not applied to validation_data")

        stats.close()

data = pd.concat(all_recordings, ignore_index=True)
stats = open("../../" + Storage_folder + "/spatial_statistics.txt","w") 

print("writing statistics:")
stats.write("\n")
stats.write("\n")
stats.write("======================================== DATA STATISTICS ========================================")
stats.write("\n")
stats.write("\n")
stats.write("Training data: " + str(training_sets))
stats.write("\n")
stats.write("\n")
stats.write("---------------------------------------- Before filtering -------------------------------------------")
stats.write("\n")
stats.write("\n")

name = "before filtering"
print("Direction")
print_column_based_statistics(data, "Direction", name=name)
print("TL_state")
print_column_based_statistics(data, "TL_state", name=name)
print("ohe_speed_limit")
print_column_based_statistics(data, speed_limit_rep, name=name)

#print("brake")
#print_brake_statistics(data, name)
#print("throttle")
#print_throttle_statistics(data, name)
#print("brake vs throttle")
#print_speed_based_on_brake_and_throttle(data, name)
print("print_steering_based_on_direction_and_speed_limit")
print_steering_based_on_direction_and_speed_limit(data, name)
print("steering")
print_steering_statistics(data, conf, name)

print("plotting")
fig, ax1 = plt.subplots()
ax1.hist(data.Steer.values, steer_bars)
fig.savefig("../../" + Storage_folder + "/steer_before_spatial_filtering")

fig, ax = plt.subplots()
ax.hist(data.Throttle.values, throttle_bars)
fig.savefig("../../" + Storage_folder + "/throttle_before_spatial_filtering")

if dataset != "Validation_data":
    data = filter_input_based_on_steering(data, conf)
    data = filter_input_based_on_not_moving(data, conf)
    #data = filter_corrupt_input(data)
    
    print("writing statistics")
    stats.write("\n")
    stats.write("\n")
    stats.write("\n")

    stats.write("---------------------------------------- After filtering -------------------------------------------")
    stats.write("\n")
    stats.write("\n")
    stats.write("\n")
    stats.write("Filtered with: ")
    stats.write("Threshold steering: " + str(conf.filter_threshold))
    stats.write("Degree steering: " + str(conf.filtering_degree))
    stats.write("Threshold speed: " + str(conf.filter_threshold_speed))
    stats.write("Degree speed(standing_still): " + str(conf.filtering_degree_speed))
    stats.write("\n")
    stats.write("\n")
    name = "after filtering"
    print("Direction")
    print_column_based_statistics(data, "Direction", name=name)
    print("TL_state")
    print_column_based_statistics(data, "TL_state", name=name)
    print("ohe_speed_limit")
    print_column_based_statistics(data, speed_limit_rep, name=name)

    #print_brake_statistics(data, name)
    #print_throttle_statistics(data, name)
    #print_speed_based_on_brake_and_throttle(data, name)
    print("print_steering_based_on_direction_and_speed_limit")
    print_steering_based_on_direction_and_speed_limit(data, name)
    print("steering")
    print_steering_statistics(data, conf, name)
    print("plotting")

    fig, ax = plt.subplots()
    ax.hist(data.Steer.values, steer_bars)
    fig.savefig("../../" + Storage_folder + "/steer_after_spatial_filtering")

    fig, ax = plt.subplots()
    ax.hist(data.Throttle.values, throttle_bars)
    fig.savefig("../../" + Storage_folder + "/throttle_after_spatial_filtering")
else:
    print("Filtering not applied to validation_data")

stats.close()