"""Updates the measurment file """
from __future__ import print_function


import os
import pandas as pd
import numpy as np
from Misc.misc import get_data_paths, get_image_name
from Misc.preprocessing import get_one_hot_encoded
from Spatial.data_configuration import Config

DATA_PATHS = get_data_paths("Training_data_temp")
start_index = 0
for path in DATA_PATHS[start_index:]:
    images = get_data_paths(path[6:] + "/Images/", sort=False)
    df = pd.read_csv(path + "/Measurments/recording.csv")
    """
    # Check if there is a jump in timeframes
    previous = 0
    for i, row in df.iterrows():
        current = int(row["frame"])
        if current - previous > 6 and previous != 0:
            print(path)
            print(current)
            print(current - previous)
        previous = current
    """
    print(len(images))
    if len(images) != len(df.index) or len(images) < 50:
        print(path)
        print("Images: " + str(len(images)))
        print("Measurments: " + str(len(df.index)))

        raw_input()
