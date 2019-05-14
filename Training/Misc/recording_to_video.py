import cv2
import numpy as np
import glob
import pandas as pd
import os
from Misc.misc import get_image
from Spatial.data_configuration import Config
CONF = Config()
MEASURMENT_PATH = "/Measurments/recording.csv"
DATA_PATHS = []
#PATH = '../../Training_data'
PATH = '../../Validation_data'
for folder in os.listdir(PATH):
    DATA_PATHS.append(PATH + "/" + folder)


for path in DATA_PATHS:
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
    out = cv2.VideoWriter(path + "/project.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
