""" Module that tests multiple models at multiple tracks """
#pylint: disable=invalid-name
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import os
import argparse
from autonomous_driver_trimmed import game_loop

# Which model architecture to use
#model_type = "Spatial"
#model_type = "Spatiotemporal"
model_type = "Temporal"
models_path = "Training/" + model_type + "/Stored_models/"
#models_path = "transfer/Stored_models" + model_type + "/"

# Which models to test
all_models = False
chosen_folders = [28]

# Which checkpoints to test
train_loss = False
val_loss = True
best_checkpoint_only = False

test_only_epoch = 0
every_n_epoch = 1
threshold_epoch = 0 # Only applicable if best_checkpoint_only is false

# Define which tracks to test
# Town 1 has 4 possible tracks
# Town 2 has 3 possible tracks
town = 2
tracks_town_1 = [0]
tracks_town_2 = [1]


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

for path in data_paths:
    checkpoints = get_paths(path + "/Checkpoints", "-", 1)
    checkpoints.reverse()
    best_val_checkpoints = [c for c in checkpoints if "val_loss" in c]
    best_train_checkpoints = [c for c in checkpoints if "train_loss" in c]
    if train_loss:
        for checkpoint in best_train_checkpoints:
            if best_checkpoint_only:
                best_val_models.append(checkpoint)
                break
            elif test_only_epoch != 0:
                 if int(checkpoint.split("-")[1]) == test_only_epoch:
                    best_val_models.append(checkpoint)
            else:
                if int(checkpoint.split("-")[1]) > threshold_epoch:
                    best_val_models.append(checkpoint)
    if val_loss:
        for checkpoint in best_val_checkpoints:
            if best_checkpoint_only:
                best_val_models.append(checkpoint)
                break
            elif test_only_epoch != 0:
                 if int(checkpoint.split("-")[1]) == test_only_epoch:
                    best_val_models.append(checkpoint)
            else:
                if int(checkpoint.split("-")[1]) > threshold_epoch:
                    best_val_models.append(checkpoint)

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
    [37, 81],
    [12, 6],
    [46, 45]
]
if town == 1:
    waypoints = waypoints_town_1
    tracks = tracks_town_1
else:
    waypoints = waypoints_town_2
    tracks = tracks_town_2
i = 0
for model in best_val_models:
    if i % every_n_epoch == 0:
        for track in tracks:
            args = argparser.parse_args()
            args.traffic_light = 0
            args.cars = 0
            args.random_weather = 0
            args.model = model
            args.model_type = model_type
            args.waypoints = waypoints[track]
            args.path = "Test_recordings/" + model_type
            game_loop(args)
    i += 1

