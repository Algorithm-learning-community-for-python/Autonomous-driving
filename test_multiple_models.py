""" Module that tests multiple models at multiple tracks """
#pylint: disable=invalid-name
import sys
sys.path.append("Training")
import os
import argparse
import pandas as pd
import numpy as np
from autonomous_driver_trimmed import game_loop
from Training.Misc.rate_test_results import rate_recording

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# Which model architecture to use
model_type = "Temporal"
models_path = "Training/" + model_type + "/Stored_models/"

# Which models to test
all_models = False
chosen_folders = [34]

# Which checkpoints to test
train_loss = False
val_loss = True
best_checkpoint_only = False
test_only_epoch = [26]
every_n_epoch = 1
threshold_epoch = 0 # Only applicable if best_checkpoint_only is false

# Define which tracks to test
# Town 1 has 4 possible tracks
# Town 2 has 3 possible tracks
town = 2
tracks_town_1 = [0]
tracks_town_2 = [-3]# range(28)


def get_paths(dir_path, split_on, sort_idx):
    """ Returns a sorted list of paths to the folders located in dir_path"""
    paths = []
    for f in os.listdir(dir_path):
        if f == ".DS_Store" or f == "store.h5":
            continue
        paths.append(dir_path + "/" + f)
    paths.sort(key=lambda a: int(a.split(split_on)[sort_idx]))
    return paths


# Fetch the models to test
if all_models:
    data_paths = get_paths(models_path, "/", -1)
else:
    data_paths = []
    for folder in chosen_folders:
        data_paths.append(models_path + str(folder))

best_loss_models = []
best_val_models = []

for i, path in enumerate(data_paths):
    checkpoints = get_paths(path + "/Checkpoints", "-", 1)
    checkpoints.reverse()
    best_val_checkpoints = [c for c in checkpoints if "val_loss" in c]
    best_train_checkpoints = [c for c in checkpoints if "train_loss" in c]
    if train_loss:
        for checkpoint in best_train_checkpoints:
            if best_checkpoint_only:
                best_val_models.append(checkpoint)
                break
            elif len(test_only_epoch) >= 1:
                if len(chosen_folders)> 1 and int(checkpoint.split("-")[1]) in test_only_epoch[i]:
                    best_val_models.append(checkpoint)
                elif len(chosen_folders) == 1 and  int(checkpoint.split("-")[1]) in test_only_epoch:
                    best_val_models.append(checkpoint)

            else:
                if int(checkpoint.split("-")[1]) > threshold_epoch:
                    best_val_models.append(checkpoint)
    if val_loss:
        for checkpoint in best_val_checkpoints:
            if best_checkpoint_only:
                best_val_models.append(checkpoint)
                break
            elif len(test_only_epoch) >= 1:
                if len(chosen_folders)> 1 and int(checkpoint.split("-")[1]) in test_only_epoch[i]:
                    best_val_models.append(checkpoint)
                elif len(chosen_folders) == 1 and  int(checkpoint.split("-")[1]) in test_only_epoch:
                    best_val_models.append(checkpoint)
            else:
                if int(checkpoint.split("-")[1]) > threshold_epoch:
                    best_val_models.append(checkpoint)


def get_results(network, model, folder):
    PATH = "Test_recordings/" + network + "/" + "model-" + str(model) + "/" + str(folder) + "/Measurments/recording.csv"
    measurements = pd.read_csv(PATH)
    steering_sum = measurements.Steer.abs().sum()
    speed_score = 0
    speed_below_score = 0
    speed_above_score = 0
    count_above = 1
    count_below = 1
    crossed_broke = 0
    crossed_none = 0
    crossed_others = 0
    collision = False
    for j, row in measurements.iterrows():
        if (row["TL_state"] == "Green"):
            speed_limit = float(row["speed_limit"])*100
            speed = float(row["Speed"])*100
            #speed_score += abs(speed-speed_limit)
            if speed < speed_limit:
                speed_below_score += speed_limit - speed
                count_below += 1
            else:
                speed_above_score += speed - speed_limit
                count_above += 1
        """
        if pd.notna(row["Lane_Invasion"]):
            crossed_type = row["Lane_Invasion"].split(" ")[-1].replace("'", "")
            if crossed_type == "Broken":
                crossed_broke += 1
            elif crossed_type == "NONE":
                crossed_none += 1
            else:
                print("New crossed-type: " + crossed_type)
                crossed_others += 1
        """
        #if row.Collision:
        #    collision = True

    steering_score = (steering_sum / len(measurements.index))*100
    #speed_score = np.round((speed_score / len(df.index)), 2)
    crossed_lanes = measurements.Lane_Invasion.count()
    speed_above_score = np.round((speed_above_score /count_above), 2)

    return steering_score, speed_above_score, crossed_lanes
# Define arguments for the carla client

argparser = argparse.ArgumentParser(
    description='CARLA Manual Control Client')
argparser.add_argument(
    '--path',
    default='Test_recordings',
    help='Where to store data')
argparser.add_argument(
    '-v', '--verbose',
    action='store_true',
    dest='debug',
    help='print debug information')
argparser.add_argument(
    '--host',
    metavar='H',
    default='127.0.0.1',
    help='IP of the host server (default: 127.0.0.1)')
argparser.add_argument(
    '-p', '--port',
    metavar='P',
    default=2000,
    type=int,
    help='TCP port to listen to (default: 2000)')
argparser.add_argument(
    '-w', '--random_weather',
    metavar='W',
    default=1,
    type=int,
    help='set to 0 use clear noon every time')
argparser.add_argument(
    '-c', '--cars',
    metavar='W',
    default=1,
    type=int,
    help='set to 0 to not include cars')
argparser.add_argument(
    '-t', '--traffic_light',
    metavar='W',
    default=1,
    type=int,
    help='set to 0 to ignore traffic lights')


# Test the models

waypoints_town_1 = [
    #[181, 24],
    [150, 24],
    [200, 70],
    [49, 209],
    [60, 27],
]
waypoints_town_2 = [
    [49, 45], #low speed straight
    [18, 25], #high speed straight
    [57, 76], #intersection left
    [46, 72], #intersection right
    [40, 62], #turn and intersection right
    [63, 39], #intersection and turn left
    #[59, 52], #circle inner city left turns
    #[54, 60], #circle inner city right turns
    [43, 68], #circle inner and outer right turns
    [35, 69], #circle inner and outer left turns
    [44, 57], #right to left turn
    [66, 46], #left to right turn
    [64, 11], #high speed with turns
    [9, 75], #high speed with turns opposite
    [64, 36], # outer track 1 
    [37, 75], # outer track 1 opposite way
    [38, 11], # outer track 2
    [9, 39], # outer track 2 opposite way
    [82, 5], # Through the city 1 long
    [7, 32], # Through the city 2 long opposite
    [70, 56], # inner city short 1
    [66, 71], # inner city short 1 opposite
    [44, 76], #inner city short 2
    [72, 45], # inner city short 2 opposite
    [75, 71], # outer to inner city 1 difficult
    [9, 45], # outer to inner city 1 difficult opposite
    [37, 63], #outer to inner city 2 difficult
    [37, 81], # Through the city 2
    [12, 6],  # High speed then through the city
    [46, 45], # inner city easy
]
if town == 1:
    waypoints = waypoints_town_1
    tracks = tracks_town_1
else:
    waypoints = waypoints_town_2
    tracks = tracks_town_2


results_folders = []
file_names = []
stop_conditions = []
conditions = []
multi_file_name = "multi_test_model-" + str(chosen_folders[0])
current_model_nr = int(best_val_models[0].split("/")[3])
try:
    os.mkdir("Test_recordings/" + model_type +"/model-" + str(current_model_nr))
except FileExistsError as fe:
    print(fe)
args = argparser.parse_args()

type_of_weathers = 1
cars = 1

# Create the pandas DataFrame 
df = pd.DataFrame(columns=['Network', 'Model', 'Result_folder', 'Epoch', 'Loss', 'Track', 'Weather', 'Cars', 'Distance', 'Steer', 'Speed', 'Completion', 'Crossed_line_count'])
  
for model in best_val_models:
    model_nr = int(model.split("/")[3])
    epoch = int(model.split("/")[-1].split("-")[1])
    loss = float(".".join(model.split("/")[-1].split("-")[2].split(".")[:2]))
    for c in range(cars):
        for w in range(type_of_weathers):
            for track in tracks:
                print("Testing model " + str(model_nr) + " epoch " + str(epoch))
                print("Testing track " + str(track) + " with weather " + str(w) + " and cars " + str(c))
                print("Waypoints: " + str(waypoints[track]))
                #store previous model results if a new model is to be tested
                if model_nr != current_model_nr:
                    df.to_csv("Test_recordings/" + model_type + "/model-" + str(current_model_nr) + "/results.csv", index=False)
                    df = pd.DataFrame(columns=['Network', 'Model', 'Result_folder', 'Epoch', 'Loss', 'Track', 'Weather', 'Cars', 'Distance', 'Steer', 'Speed', 'Completion', 'Crossed_line_count'])

                    rate_recording(stop_conditions=stop_conditions, path=args.path, cur_folder=results_folders, file_name=file_names,
                                    multi_rating=True, multi_file_name=multi_file_name, conditions=conditions)
                    results_folders, file_names, stop_conditions, conditions = [], [], [], []
                    multi_file_name = "multi_test_model-" + str(model_nr)
                    current_model_nr = model_nr
                    os.mkdir("Test_recordings/" + model_type + "/model-" + str(current_model_nr))

                conditions.append({"traffic_lights": c, "cars": c, "weather": w, "track": track})
                args = argparser.parse_args()
                args.traffic_light = c
                args.cars = c
                args.random_weather = w
                args.track = track

                args.model = model
                args.model_type = model_type
                args.waypoints = waypoints[track]
                args.path = "Test_recordings/" + model_type + "/model-" + str(current_model_nr)
                stop_condition, results_folder, file_name, distance = game_loop(args)
                if stop_condition != "not_far_enough":

                    steer, speed, crossed_lines_count = get_results(model_type, current_model_nr, results_folder)
                    stop_conditions.append(stop_condition)
                    results_folders.append(results_folder)
                    file_names.append(file_name)
                else:
                    steer = None
                    speed = None
                    crossed_lines_count = None
                temp_df = pd.DataFrame({
                    'Network': model_type,
                    'Model':model_nr,
                    'Result_folder': results_folder,
                    'Epoch': epoch,
                    'Loss': loss,
                    'Track': track,
                    'Weather': w,
                    'Cars': c,
                    'Distance': distance,
                    'Steer': steer,
                    'Speed': speed,
                    'Completion': stop_condition,
                    'Crossed_line_count': crossed_lines_count
                }, index=[0])
                df = df.append(temp_df)

df.to_csv("Test_recordings/" + model_type + "/model-" + str(current_model_nr) + "/results.csv", index=False)

# Rate the last recording
rate_recording(
    stop_conditions=stop_conditions,
    path=args.path,
    cur_folder=results_folders,
    file_name=file_names,
    multi_rating=True,
    multi_file_name=multi_file_name,
    conditions=conditions
)