
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
#for folder in os.listdir(PATH):
#    DATA_PATHS.append(PATH + "/" + folder)


for path in p:
    print(path)
    for filename in os.listdir(path):
        #print(filename)
        if ".avi" in filename:
            #print(filename)
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
                    cv2.resizeWindow('Frame', 920,690)

                    cv2.imshow('Frame', frame)

                    # Press Q on keyboard to  exit
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                
                # Break the loop
                else:
                    break
            
            # When everything done, release  
            # the video capture object 
            cap.release() 
            
            # Closes all the frames 
            cv2.destroyAllWindows() 