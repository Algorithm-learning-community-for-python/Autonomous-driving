""" Module that prints relevant statistics """
#pylint: disable=invalid-name
#pylint: disable=superfluous-parens
import pandas as pd
from preprocessing import (
    filter_sequence_input_based_on_steering, \
    filter_sequence_input_based_on_not_moving, \
    filter_corrupt_sequence_input, \
    )
from Spatiotemporal.data_configuration import Config
import os
import matplotlib
matplotlib.use("agg")

from misc import get_data_paths
import matplotlib.pyplot as plt
pd.set_option("display.max_rows", 500)

def print_column_based_statistics(df, col, name="", count=True, mean=True, std=False):

    total_samples = len(df.index)
    df_grouped = df.groupby(col)

    stats.write("############################ "+ col.upper() +" BASED STATISTICS " + name + " ############################")
    stats.write("\n")
    stats.write("\n")

    if count:
        stats.write("############## Sample count " + name + "##############")
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
        stats.write("############## Mean values " + name + " ##############")
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
        stats.write("############## Standard deviation " + name + " ##############")
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
    stats.write("############################ "+ target.upper() +" STATISTICS " + name + " ############################")
    stats.write("\n")

    stats.write("############## based on "+ groups +" ##############")
    stats.write("\n")
    stats.write("\n")
    
    if count:
        samples_per_direction = df_grouped.count()
        stats.write("############## Sample count " + name + " ##############")
        stats.write("\n")

        stats.write("Total samples per " + groups + ": ")
        stats.write("\n")

        stats.write(str(samples_per_direction["frame"]))
        stats.write("\n")
        stats.write("\n")

    if mean: 
        mean_per_direction = df_grouped.mean()
        stats.write("############## Mean values " + name + " ##############")
        stats.write("\n")

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
    print_multi_column_based_statistics(df, ["TL_state", "Speed_binned"], "Brake", name="")
    stats.write("\n")
    stats.write("\n")
    stats.write("\n")

def print_throttle_statistics(df, name=""):
    df["Speed_binned"] = pd.cut(df["Speed"], 10)
    print_multi_column_based_statistics(df, ["TL_state", "Speed_binned"], "Throttle", name="") 
    stats.write("\n")
    stats.write("\n")
    stats.write("\n")

def print_brake_vs_throttle(df, name=""):
    df["Throttle_binned"] = pd.cut(df["Throttle"], 10)
    df["Brake_binned"] = pd.cut(df["Brake"], 10)
    print_multi_column_based_statistics(df, ["Throttle_binned", "Brake_binned"], "Speed", name="")
    stats.write("\n")
    stats.write("\n")
    stats.write("\n")



def print_steering_statistics(df, conf, name=""):
    df["Steer_binned"] = pd.cut(df["Steer"], 10)
    threshold = conf.filter_threshold
    print_multi_column_based_statistics(df, ["Direction", "Steer_binned"], "frame", name="", mean=False, std=False)

    stats.write("############## RANGE OF STEERING SAMPLES " + name + " ##############")
    stats.write("\n")

    l = len([(s, d) for (s,d) in zip(df.Steer.values, df.Direction.values) if abs(s) < threshold and (d =="[0. 0. 1. 0. 0. 0. 0.]" or d =="[0. 0. 0. 0. 0. 0. 1.]")])
    stats.write("Samples steering less than " + str(threshold) + " and lanefollow: " + str(l))
    stats.write("\n")

    l = len([x for x in df.Steer.values if abs(x) > threshold])
    stats.write("Samples steering more than " + str(threshold) + ": " + str(l))
    stats.write("\n")

    l = len([x for x in df.Steer.values if abs(x) > 0.4])
    stats.write("Samples steering more than 0.4: " + str(l))
    stats.write("\n")
    stats.write("\n")
    stats.write("\n")

conf = Config()
data_paths=[]
recordings_path = conf.recordings_path
if recordings_path == "/Measurments/recording.csv":
    speed_limit_rep = "speed_limit"
else:
    speed_limit_rep = "ohe_speed_limit"

temporal = False
dataset = "Training_data"
#dataset = "Validation_data"

print("Fetching data_paths")
data_paths = get_data_paths(dataset)
print("Fetched " + str(len(data_paths)) + " episodes")
data = []
step_size = conf.step_size_training
skip_steps = conf.skip_steps
seq_len = conf.input_size_data["Sequence_length"]
skipped_samples = 0
non_skipped_turn = 0
not_skipped = 0
# Use subset avoid using all the data
percentage_of_training_data = 0.05
subset = int(len(data_paths)*percentage_of_training_data)

print("Fetching data from the first " + str(subset) + " episodes")
for r, path in enumerate(data_paths[:subset]):
    df = pd.read_csv(path + conf.recordings_path)
    df["Images_path"] = path + "/Images/"
    for i in range(0, len(df)):
        if i + (seq_len*step_size) < len(df):
            indexes = []
            lanefollow = True
            for j in range(i, i + (seq_len*step_size), step_size):
                if df.iloc[j, :].Direction != "[0. 0. 1. 0. 0. 0. 0.]" and df.iloc[j, :].Direction != "[0. 0. 0. 0. 0. 0. 1.]":
                    lanefollow = False
                indexes.append(j)
            if i % skip_steps == 0:
                not_skipped += 1
                data.append(df.iloc[indexes, :].copy())
            elif not lanefollow:
                non_skipped_turn += 1
                data.append(df.iloc[indexes, :].copy())
            else:
                skipped_samples += 1

print(str(len(data)) + " datasamples gathered")
print(str(skipped_samples) + " samples not added due to skip_steps")
sequence_data = data
target_data = []
for sequence in data:
    target_data.append(sequence.iloc[-1, :])

data = pd.DataFrame(target_data)
if True:
    print("writing statistics:")
    stats = open(dataset + "_statistics.txt","w")
    stats.write("\n")
    stats.write("\n")
    stats.write("======================================== DATA STATISTICS ========================================")
    stats.write("\n")
    stats.write("\n")

    stats.write("---------------------------------------- BEFORE FILTERING -------------------------------------------")
    stats.write("\n")
    stats.write("\n")

    name = "BEFORE FILTERING"
    print("Direction")
    print_column_based_statistics(data, "Direction", name=name)
    print("TL_state")
    print_column_based_statistics(data, "TL_state", name=name)
    print("ohe_speed_limit")
    print_column_based_statistics(data, "ohe_speed_limit", name=name)

    print("brake")
    print_brake_statistics(data, name)
    print("throttle")
    print_throttle_statistics(data, name)
    print("brake vs throttle")
    print_brake_vs_throttle(data, name)
    print("steering")
    print_steering_statistics(data, conf, name)

    print("plotting")
    fig1, ax1 = plt.subplots()
    ax1.hist(data.Steer.values, 20)
    fig1.savefig("before_filtering")

if dataset == "Validation_data":
    stats.close()
    print("Filtering not applied to validation_data")
    exit()

sequence_data = filter_sequence_input_based_on_steering(sequence_data, conf)
sequence_data = filter_sequence_input_based_on_not_moving(sequence_data, conf)
sequence_data = filter_corrupt_sequence_input(sequence_data)

target_data = []
for sequence in sequence_data:
    target_data.append(sequence.iloc[-1,: ])
data = pd.DataFrame(target_data)

print("writing statistics")
stats.write("\n")
stats.write("\n")
stats.write("\n")

stats.write("---------------------------------------- AFTER FILTERING -------------------------------------------")
stats.write("\n")
stats.write("\n")
stats.write("\n")


name = "AFTER FILTERING"
print("Direction")
print_column_based_statistics(data, "Direction", name=name)
print("TL_state")
print_column_based_statistics(data, "TL_state", name=name)
print("ohe_speed_limit")
print_column_based_statistics(data, "ohe_speed_limit", name=name)

print("brake")
print_brake_statistics(data, name)
print("throttle")
print_throttle_statistics(data, name)
print("brake vs throttle")
print_brake_vs_throttle(data, name)
print("steering")
print_steering_statistics(data, conf, name)

print("plotting")
fig2, ax2 = plt.subplots()
ax2.hist(data.Steer.values, 20)
fig2.savefig("after_filtering")

stats.close()
