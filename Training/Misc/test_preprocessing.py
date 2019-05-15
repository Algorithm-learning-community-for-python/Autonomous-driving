import pandas as pd
from preprocessing import filter_input_based_on_steering
from Spatial.data_configuration import Config
import os
import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
from misc import get_data_paths

conf = Config()
data_paths=[]
temporal = False
data_paths = get_data_paths()
recordings = []
if temporal:
    store = pd.HDFStore("../../Training_data/store.h5")
    for path in data_paths:
        df_name = "Recording_" + path.split("/")[-1]
        recording = store[df_name]
        recordings.append(recording)
else:
    for path in data_paths:
        df = pd.read_csv(path+"/Measurments/recording.csv")
        
        recordings.append(df)
        

follow_lane = 0
intersection = 0
total_samples = 0
steerings = []
for dataframe in recordings:
    for _, row in dataframe.iterrows():
        # Add or remove the current sequence
        follow = True
        if temporal:
            directions = row["Direction"]
            steerings.append(row["Steer"][-1])
            for direction in directions:
                if direction[2] == 0:
                    follow = False
        else:
            steerings.append(row["Steer"])
            if row["Direction"]!="RoadOption.LANEFOLLOW" and row["Direction"]!="":
                follow = False

        total_samples += 1
        if follow:
            follow_lane += 1
        else:
            intersection += 1
print("BEFORE")
print("Total samples: " + str(total_samples))
print("intersections: " + str(intersection))
print("Follow lane: " + str(follow_lane))
l = len([x for x in steerings if abs(x) > 0.1])
print("Samples steering more than 0.1: " + str(l))
l = len([x for x in steerings if abs(x) > 0.4])
print("Samples steering more than 0.4: " + str(l))
fig1, ax1 = plt.subplots()
ax1.hist(steerings, 20)
fig1.savefig("before_filtering")

follow_lane = 0
intersection = 0
total_samples = 0
steerings = []
for recording in recordings:
    dataframe = filter_input_based_on_steering(recording, conf, temporal)
    for _, row in dataframe.iterrows():
        # Add or remove the current sequence
        follow = True
        if temporal:
            directions = row["Direction"]
            steerings.append(row["Steer"][-1])
            for direction in directions:
                if direction[2] == 0:
                    follow = False
        else:
            steerings.append(row["Steer"])
            if row["Direction"]!="RoadOption.LANEFOLLOW" and row["Direction"]!="":
                follow = False

        total_samples += 1
        if follow:
            follow_lane += 1
        else:
            intersection += 1


print("AFTER")
print("Total samples: " + str(total_samples))
print("intersections: " + str(intersection))
print("Follow lane: " + str(follow_lane))
l = len([x for x in steerings if abs(x) > 0.1])
print("Samples steering more than 0.1: " + str(l))
l = len([x for x in steerings if abs(x) > 0.4])
print("Samples steering more than 0.4: " + str(l))
fig2, ax2 = plt.subplots()
ax2.hist(steerings, 20)
fig2.savefig("after_filtering")