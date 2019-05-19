import cv2
import numpy as np
import glob
import pandas as pd
import os
from Misc.misc import get_image
from Spatial.data_configuration import Config

def recording_to_video(path="../../Test_recordings", cur_folder=None, file_name=None):
    CONF = Config()
    MEASURMENT_PATH = "/Measurments/recording.csv"
    DATA_PATHS = []
    #PATH = '../../Training_data'
    #PATH = '../../Validation_data'
    PATH = path 
    if cur_folder == None:
        for folder in os.listdir(PATH):
            if folder == "store.h5":
                continue
            DATA_PATHS.append(PATH + "/" + folder)
    else:
        DATA_PATHS.append(PATH + "/" + str(cur_folder))

    for path in DATA_PATHS:
        print(cur_folder)
        print(path)
        df = pd.read_csv(path + MEASURMENT_PATH)
        img_array = []
        for j, row in df.iterrows():
            cur_frame = row["frame"]
            img = get_image(path + "/Images/", cur_frame)
            # Cropping
            img = img[CONF.top_crop:, :, :]
            img_array.append(img)

        height, width, layers = img_array[0].shape
        size = (width,height)
        if file_name == None:
            name = "/project.avi"
        else:
            name = "/" + file_name.replace("/", "-") + ".avi"
        out = cv2.VideoWriter(path + name, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

#recording_to_video('../../Training_data')
#recording_to_video(path="../../Test_recordings", cur_folder=11)
