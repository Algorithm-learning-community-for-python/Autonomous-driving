import pandas as pd
from preprocessing import filter_input_based_on_steering, filter_input_based_on_speed_and_tl, filter_corrupt_input
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



def print_steering_statistics(df, name=""):
    df["Steer_binned"] = pd.cut(df["Steer"], 10)
    print_multi_column_based_statistics(df, ["Direction", "Steer_binned"], "frame", name="", mean=False, std=False)

    stats.write("############## RANGE OF STEERING SAMPLES " + name + " ##############")
    stats.write("\n")

    l = len([(s, d) for (s,d) in zip(df.Steer.values, df.Direction.values) if abs(s) < 0.1 and d =="RoadOption.LANEFOLLOW"])
    stats.write("Samples steering less than 0.1 and lanefollow: " + str(l))
    stats.write("\n")

    l = len([x for x in df.Steer.values if abs(x) > 0.1])
    stats.write("Samples steering more than 0.1: " + str(l))
    stats.write("\n")

    l = len([x for x in df.Steer.values if abs(x) > 0.4])
    stats.write("Samples steering more than 0.4: " + str(l))
    stats.write("\n")
    stats.write("\n")
    stats.write("\n")

conf = Config()
data_paths=[]
temporal = False
#dataset = "Training_data"
dataset = "Validation_data"

data_paths = get_data_paths(dataset)
recordings = []
if temporal:
    store = pd.HDFStore("../../Training_data/store.h5")
    for path in data_paths:
        df_name = "Recording_" + path.split("/")[-1]
        recording = store[df_name]
        recordings.append(recording)
else:
    for path in data_paths:
        df = pd.read_csv(path + "/Measurments/recording.csv")
        #for i, row in df.iterrows():
            #if row["Direction"] == "RoadOption.LEFT" and row["Steer"] > 0.9:
                #print(row)
                #print(path)
        
        recordings.append(df)
    data = pd.concat(recordings, ignore_index=True)
#exit()


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
print_column_based_statistics(data, "Direction", name=name)
print_column_based_statistics(data, "TL_state", name=name)
print_column_based_statistics(data, "speed_limit", name=name)

print_brake_statistics(data, name)
print_throttle_statistics(data, name)
print_brake_vs_throttle(data, name)
print_steering_statistics(data, name)

fig1, ax1 = plt.subplots()
ax1.hist(data.Steer.values, 20)
fig1.savefig("before_filtering")

if dataset == "Validation_data":
    stats.close()
    print("Filtering not applied to validation_data")
    exit()

data = filter_input_based_on_steering(data, conf, temporal=False)
data = filter_input_based_on_speed_and_tl(data, conf, temporal=False)
data = filter_corrupt_input(data, conf, temporal=False)
stats.write("\n")
stats.write("\n")
stats.write("\n")

stats.write("---------------------------------------- AFTER FILTERING -------------------------------------------")
stats.write("\n")
stats.write("\n")
stats.write("\n")


name = "AFTER FILTERING"
print_column_based_statistics(data, "Direction", name=name)
print_column_based_statistics(data, "TL_state", name=name)
print_column_based_statistics(data, "speed_limit", name=name)

print_brake_statistics(data, name)
print_throttle_statistics(data, name)
print_brake_vs_throttle(data, name)
print_steering_statistics(data, name)

fig2, ax2 = plt.subplots()
ax2.hist(data.Steer.values, 20)
fig2.savefig("after_filtering")

stats.close()
