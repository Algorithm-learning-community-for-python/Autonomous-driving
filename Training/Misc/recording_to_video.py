import cv2
import numpy as np
import glob
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from Misc.misc import get_image
from Spatiotemporal.data_configuration import Config

def recording_to_video(path="../../Test_recordings", cur_folder=None, file_name=None, use_hsv=False, crop=False):
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
        print(path)
        df = pd.read_csv(path + MEASURMENT_PATH)
        img_array = []
        hsv_array = []
        for j, row in df.iterrows():
            cur_frame = row["frame"]
            try:
                img = get_image(path + "/Images/", cur_frame)
                if crop:
                    img_size = CONF.input_size_data["Image"]
                    img = img[CONF.top_crop:, :, :]
                    img = cv2.resize(img,(img_size[1], img_size[0]))

                if use_hsv:
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                speed_limit = int(float(row["speed_limit"])*100)
                speed = int(float(row["Speed"])*100)
                if pd.notna(row["Direction"]):
                    direction = row["Direction"]
                    direction = direction.split(".")[1]
                else:
                    direction = "LANEFOLLOW"


                tl = row["TL_state"]
                cv2.putText(img,"Speed: " + str(speed) + " / " + str(speed_limit), 
                    (10,280), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4,
                    (0,0,0),
                    1)

                cv2.putText(img, "Traffic light: " + tl, 
                    (10,295), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4,
                    (0,0,0),
                    1)

                cv2.putText(img, "Direction: " + direction, 
                    (10,310), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4,
                    (0,0,0),
                    1)
                if use_hsv:
                    cv2.putText(hsv,"Speed: " + str(speed) + " / " + str(speed_limit), 
                        (10,280), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4,
                        (0,0,0),
                        1)

                    cv2.putText(hsv, "Traffic light: " + tl, 
                        (10,295), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4,
                        (0,0,0),
                        1)

                    cv2.putText(hsv, "Direction: " + direction, 
                        (10,310), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4,
                        (0,0,0),
                        1)
                #cv2.imshow('image',img)
                #cv2.waitKey(0)
                # Cropping
                #img = img[CONF.top_crop:, :, :]

                img_array.append(img)
                if use_hsv:
                    hsv_array.append(hsv)
            except TypeError:
                continue
            
        try:
            height, width, layers = img_array[0].shape
        except IndexError as i:
            print("No images found at " + path)
            print(i)
            continue
        size = (width,height)
        if file_name == None:
            name = "/project.avi"
            name2 = "/project_hsv.avi"
        else:
            name = "/" + file_name.replace("/", "-") + ".avi"
            name2 = "/" + file_name.replace("/", "-") + "_hsv.avi"
        out = cv2.VideoWriter(path + name, cv2.VideoWriter_fourcc(*'DIVX'), 20, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        if use_hsv:
            out = cv2.VideoWriter(path + name2, cv2.VideoWriter_fourcc(*'DIVX'), 20, size)
            for i in range(len(img_array)):
                out.write(hsv_array[i])
            out.release()

#recording_to_video('../../Training_data_temp', cur_folder=45, use_hsv=False, crop=False)
#recording_to_video(path="../../Test_recordings", cur_folder=74, use_hsv=True, crop=True)
