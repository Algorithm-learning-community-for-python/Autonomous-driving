import pandas as pd
from preprocessing import filter_input_based_on_steering
from data_configuration import Config
import os
import matplotlib.pyplot as plt

conf = Config()
data_paths=[]
for folder in sorted(os.listdir('../../Training_data')):
    if folder == ".DS_Store" or folder == "store.h5":
        continue
    data_paths.append("../../Training_data/" + folder)
data_paths.sort(key=lambda a: int(a.split("/")[-1]))
recordings = []
store = pd.HDFStore("../../Training_data/store.h5")
for path in data_paths:
    df_name = "Recording_" + path.split("/")[-1]
    recording = store[df_name]
    recordings.append(recording)

follow_lane = 0
intersection = 0
total_samples = 0
steerings = []
for dataframe in recordings:
    for _, row in dataframe.iterrows():
        # Add or remove the current sequence
        directions = row["Direction"]
        steerings.append(row["Steer"][-1])
        follow = True
        for direction in directions:
            if direction[2] == 0:
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
plt.hist(steerings, 20)
plt.show()

follow_lane = 0
intersection = 0
total_samples = 0
steerings = []
for recording in recordings:
    dataframe = filter_input_based_on_steering(recording)
    for _, row in dataframe.iterrows():
        # Add or remove the current sequence
        directions = row["Direction"]
        steerings.append(row["Steer"][-1])
        follow = True
        for direction in directions:
            if direction[2] == 0:
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
plt.hist(steerings, 20)
plt.show()