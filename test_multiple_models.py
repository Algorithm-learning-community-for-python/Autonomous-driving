from autonomous_driver_trimmed import game_loop
from Misc.misc import get_data_paths
import os
import argparse

def get_paths(path, split, idx):
    paths=[]
    for folder in os.listdir(path):
        if folder == ".DS_Store" or folder == "store.h5":
            continue
        paths.append(path + "/" + folder)
    paths.sort(key=lambda a: int(a.split(split)[idx]))
    return paths


models_path = "Training/Spatial/Stored_models"
data_paths = get_paths(models_path, "/", -1)

best_loss_models = []
best_val_models = []
for path in data_paths:
    checkpoints = get_paths(path + "/Checkpoints", "-", 2)
    best_loss = True
    best_val = True
    checkpoints.reverse()
    for checkpoint in checkpoints:
        if best_loss and "loss" in checkpoint:
            best_loss = False
            best_loss_models.append(checkpoint)
        if best_val and "val" in checkpoint:
            best_val = False
            best_val_models.append(checkpoint)

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


for model in best_loss_models:
    args = argparser.parse_args()
    args.model = model
    game_loop(args)

for model in best_val_models:
    args = argparser.parse_args()
    print(model)
    args.model = model
    game_loop(args)
