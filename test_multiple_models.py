from autonomous_driver_trimmed import game_loop
from Misc.misc import get_data_paths
import os
import argparse

def get_paths(path, split_on, sort_idx):
    paths=[]
    for folder in os.listdir(path):
        if folder == ".DS_Store" or folder == "store.h5":
            continue
        paths.append(path + "/" + folder)
    paths.sort(key=lambda a: int(a.split(split_on)[sort_idx]))
    return paths



all_models = False
only_best = True
best_loss_models = []
best_val_models = []
model_type = "Spatiotemporal"
models_path = "Training/" + model_type + "/Stored_models/"
if all_models:
    data_paths = get_paths(models_path, "/", -1)
    for path in data_paths:
        checkpoints = get_paths(path + "/Checkpoints", "-", 2)
        best_loss = True
        best_val = True
        checkpoints.reverse()
        for checkpoint in checkpoints:
            #if best_loss and "loss" in checkpoint:
            #    best_loss = False
            #    best_loss_models.append(checkpoint)
            if best_val and "val" in checkpoint:
                best_val = False
                best_val_models.append(checkpoint)
else:
    for cur_folder in range(5,6):
        checkpoints = get_paths(models_path + str(cur_folder) + "/Checkpoints", "-", 1)
        checkpoints.reverse()
        for checkpoint in checkpoints:
            #if "loss" in checkpoint:
            #    best_loss_models.append(checkpoint)
            if "val" in checkpoint:
                best_val_models.append(checkpoint)
                if only_best:
                    break


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

waypoints_town_1 = [
    [150,24],
    [200, 70],
    [49, 209],
    [60, 27],
]

waypoints_town_2 = [
    [30, 81],
    [12, 6],
    [46, 45]
]

waypoints = waypoints_town_1

"""
for model in best_loss_models:
    args = argparser.parse_args()
    args.model = model
    game_loop(args)
for model in best_val_models:
    args = argparser.parse_args()
    args.model = model
    args.waypoints =[-1,0]
    game_loop(args)
"""
for model in best_val_models:
    for wps in waypoints_town_1:
        args = argparser.parse_args()
        args.model = model #"Training/Spatial/Stored_models/6/Checkpoints/train_loss-06-0.097.hdf5" #model
        args.model_type = model_type
        args.waypoints = wps
        game_loop(args)

"""
for wps in waypoints_town_2:
    args = argparser.parse_args()
    args.model = model #"Training/Spatial/Stored_models/6/Checkpoints/train_loss-06-0.097.hdf5" #model
    args.waypoints = wps
    game_loop(args)

"""