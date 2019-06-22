import cv2
import numpy as np
import glob
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from Misc.misc import get_image, get_data_paths, get_image_name
from Spatiotemporal.data_configuration import Config


def update_images(dir_path="../../Training_data_temp", cur_folder=None, file_name=None):
    CONF = Config()
    MEASURMENT_PATH = "/Measurments/recording.csv"
    DATA_PATHS = []
    if cur_folder == None:
        DATA_PATHS = get_data_paths(dir_path)
    else:
        DATA_PATHS.append(dir_path + "/" + str(cur_folder))

    for path in DATA_PATHS:
        try:
            os.mkdir(path + "/Updated_images/")  
        except OSError as ose:
            print("Folder allready excists")
        df = pd.read_csv(path + MEASURMENT_PATH)
        for j, row in df.iterrows():
            cur_frame = row["frame"]
            img = get_image(path + "/Images/", cur_frame)
            img_size = CONF.input_size_data["Image"]
            img = img[CONF.top_crop:, :, :]
            img = cv2.resize(img,(img_size[1], img_size[0]))
            cv2.imwrite(path + "/Updated_images/" + get_image_name(cur_frame), img)

update_images(dir_path="Training_data_temp", cur_folder=None)