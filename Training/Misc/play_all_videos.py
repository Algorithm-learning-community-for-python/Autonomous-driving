
# importing libraries 
import cv2 
import numpy as np 
import os
   
MEASURMENT_PATH = "/Measurments/recording.csv"
DATA_PATHS = []
#PATH = '../../Training_data'
PATH = '../../Validation_data'
for folder in os.listdir(PATH):
    DATA_PATHS.append(PATH + "/" + folder)


for path in DATA_PATHS:
    # Create a VideoCapture object and read from input file 
    cap = cv2.VideoCapture(path + "/project.avi")

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video  file")
    
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        cap.set(3, 320)
        cap.set(4, 140)
        if ret == True:
        
            # Display the resulting frame

            cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Frame', 640,280)

            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else:
            break
    
    # When everything done, release  
    # the video capture object 
    cap.release() 
    
    # Closes all the frames 
    cv2.destroyAllWindows() 