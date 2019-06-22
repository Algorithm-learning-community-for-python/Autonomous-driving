
""" Plays the recordings of all the recordings """
# importing libraries 
import cv2 
import numpy as np 
import os
from Misc.misc import get_data_paths
MEASURMENT_PATH = "/Measurments/recording.csv"
PATH = '../../Training_data'
#PATH = '../../Validation_data'
#PATH = '../../Test_recordings'
p = get_data_paths("Training_data_temp")

for path in p[15:]:
    print(path.split("/")[-1])
    for filename in os.listdir(path):
        if ".avi" in filename:
            # Create a VideoCapture object and read from input file 
            cap = cv2.VideoCapture(path + "/" + filename)

            # Check if camera opened successfully
            if (cap.isOpened()== False):
                print("Error opening video  file")
            
            # Read until video is completed
            while(cap.isOpened()):
                
                # Capture frame-by-frame
                ret, frame = cap.read()
                cap.set(3, 460)
                cap.set(4, 345)
                if ret == True:
                
                    # Display the resulting frame

                    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Frame', 2560, 1440)

                    cv2.imshow('Frame', frame)

                    # Press Q on keyboard to  exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Break the loop
                else:
                    break
            
            # When everything done, release  
            # the video capture object 
            cap.release() 
            
            # Closes all the frames 
            cv2.destroyAllWindows() 