import cv2
import numpy as np
import glob
import pandas as pd
import os
from Misc.misc import get_image

def rate_recording(stop_conditions=None, path="../../Test_recordings", cur_folder=None, file_name=None, multi_rating=False, multi_file_name=None):
    # FILE NAME IS ACTUALLY MODEL INFORMATION
    model_info = file_name
    original_path = path
    MEASURMENT_PATH = "/Measurments/recording.csv"
    DATA_PATHS = []
    #PATH = '../../Training_data'
    #PATH = '../../Validation_data'
    PATH = path 
    if cur_folder is None:
        for folder in os.listdir(PATH):
            if folder == "store.h5":
                continue
            DATA_PATHS.append(PATH + "/" + folder)
    else:
        if multi_rating:
            for folder in cur_folder:
                DATA_PATHS.append(PATH + "/" + str(folder))
        else:
            DATA_PATHS.append(PATH + "/" + str(cur_folder))

    steering_scores = []
    speed_scores = []
    speed_below_scores = []
    count_belows = []
    speed_above_scores = []
    count_aboves = []

    crossed_broke_list = []
    crossed_none_list = []
    crossed_others_list = []

    time_steps = []
    model_type = []
    model_number = []
    model_epoch = []
    model_loss = []

    collisions = 0
    not_moving = 0
    targets_reached = 0
    if stop_conditions is not None:
        for stop_condition in stop_conditions:
            if stop_condition == "not_moving":
                not_moving += 1
            elif stop_condition == "target_reached":
                targets_reached += 1
            elif stop_condition == "collision":
                collisions += 1
    for path in DATA_PATHS:
        df = pd.read_csv(path + MEASURMENT_PATH)
        steering_sum = df.Steer.abs().sum()
        
        speed_score = 0
        speed_below_score = 0
        speed_above_score = 0
        count_above = 1
        count_below = 1
        crossed_broke = 0
        crossed_none = 0
        crossed_others = 0
        collision = False
        for j, row in df.iterrows():
            if (row["TL_state"] == "Green"):
                speed_limit = float(row["speed_limit"])*100
                speed = float(row["Speed"])*100
                speed_score += abs(speed-speed_limit)
                if speed < speed_limit:
                    speed_below_score += speed_limit - speed
                    count_below += 1
                else:
                    speed_above_score += speed - speed_limit
                    count_above += 1

            if pd.notna(row["Lane_Invasion"]):
                crossed_type = row["Lane_Invasion"].split(" ")[-1].replace("'", "")
                if crossed_type == "Broken":
                    crossed_broke += 1
                elif crossed_type == "NONE":
                    crossed_none += 1
                else:
                    print("New crossed-type: " + crossed_type)
                    crossed_others += 1

            if row.Collision:
                collision = True
        if multi_rating:
            steering_scores.append(int((steering_sum / len(df.index))*100))
            speed_scores.append(np.round((speed_score / len(df.index)), 2))
            speed_below_scores.append(np.round((speed_below_score / count_below), 2))
            speed_above_scores.append(np.round((speed_above_score /count_above), 2))
            count_belows.append(count_below)
            count_aboves.append(count_above)
            crossed_broke_list.append(crossed_broke)
            crossed_none_list.append(crossed_none)
            crossed_others_list.append(crossed_others)
        else:
            steering_score = int((steering_sum / len(df.index))*100)
            speed_score = np.round((speed_score / len(df.index)), 2)
            speed_below_score = np.round((speed_below_score / count_below), 2)
            speed_above_score = np.round((speed_above_score /count_above), 2)


        #1200-Training/Spatial/Stored_models/1/Checkpoints/best-val-08-0.116
        #new
        #1200-Training/Spatial/Stored_models/1/Checkpoints/val_loss-08-0.116

        if model_info == None:
            time_steps = "Missing"
            model_number = "Missing"
            model_type = "Missing"
            loss_type = "Missing"
            epoch = "Missing"
            loss = "Missing"
        elif not multi_rating:
            model_info = model_info.replace("/", "-").split("-")
            #["1200","Training", "Spatial", "Stored_models", "1", "Checkpoints", "val_loss", "08","0.116"]
            time_steps = model_info[0]
            model_type = model_info[2]
            model_number =  model_info[4]
            if model_number == "Current_model":
                loss_type = "Missing"
                epoch = "Missing"
                loss = "Missing"
            else:
                loss_type = model_info[6]
                if loss_type == "best":
                    loss_type = model_info[6] + " " + model_info[7]
                    epoch =  model_info[8]
                    loss =  model_info[9]
                else:
                    epoch =  model_info[7]
                    loss =  model_info[8]
        if not multi_rating:
            f = open(path + "/results.txt", "w")
            f.write("=========================== RECORDING RESULTS ==================================")
            f.write("\n")
            f.write("\n")
            f.write("--------------------------- Model info ---------------------------------------")
            f.write("\n")
            f.write("Time steps: " + str(time_steps))
            f.write("\n")
            f.write("Model number: " + str( model_number))
            f.write("\n")
            f.write("Model type: " + str(model_type))
            f.write("\n")
            f.write("Loss type: " + str(loss_type))
            f.write("\n")
            f.write("Epoch: " + str(epoch))
            f.write("\n")
            f.write("Loss: " + str(loss))
            f.write("\n")
            f.write("\n")
            f.write("--------------------------- Driving scores ---------------------------------------")
            f.write("\n")
            f.write("Steering score: " + str(steering_score))
            f.write("\n")
            f.write("Speed score: " + str(speed_score))
            f.write("\n")
            f.write("Speed below score: " + str(speed_below_score))
            f.write("\n")
            f.write("Speed above score: " + str(speed_above_score))
            f.write("\n")
            f.write("Count below speed_limit: " + str(count_below))
            f.write("\n")
            f.write("Count above speed_limit: " + str(count_above))
            f.write("\n")
            f.write("Crossed broke line: " + str(crossed_broke))
            f.write("\n")
            f.write('Crossed "none" line: ' + str(crossed_none))
            f.write("\n")
            f.write("crossed other lines: " + str(crossed_others))
            f.write("\n")
            f.write("collision: " + str(collision))
            f.write("\n")
            f.write("\n")
            f.close()
    if multi_rating:
        for info in model_info:
            mi = info.replace("/", "-").split("-")
            #["1200","Training", "Spatial", "Stored_models", "1", "Checkpoints", "val_loss", "08","0.116"]
            time_steps.append(int(mi[0]))
            model_type.append(mi[2])
            model_number.append(mi[4])
            if model_number == "Current_model":
                loss_type = "Missing"
                epoch = "Missing"
                loss = "Missing"
            else:
                loss_type = mi[6]
                if loss_type == "best":
                    loss_type = mi[6] + " " + mi[7]
                    epoch =  mi[8]
                    loss =  mi[9]
                else:
                    epoch =  mi[7]
                    loss =  mi[8]
            model_epoch.append(epoch)
            model_loss.append(loss)

        fname = original_path + "/" + multi_file_name + ".txt"
        if os.path.exists(fname):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not

        f = open(fname, append_write)
        f.write("=========================== RECORDING RESULTS ==================================")
        f.write("\n")
        f.write("Total episodes: " + str(len(time_steps)))
        f.write("\n")
        f.write("--------------------------- Model info ---------------------------------------")
        f.write("\n")
        f.write("Time steps: " + str(time_steps))
        f.write("\n")
        f.write("mean: " + str( float(sum(time_steps))/len(time_steps) ))
        f.write("\n")
        f.write("Model number: " + str(model_number))
        f.write("\n")
        f.write("Model type: " + str(model_type))
        f.write("\n")
        #f.write("Loss type: " + str(loss_type_list))
        #f.write("\n")
        f.write("Epoch: " + str(model_epoch))
        f.write("\n")
        f.write("Loss: " + str(model_loss))
        f.write("\n")
        f.write("\n")
        f.write("--------------------------- Driving scores ---------------------------------------")
        f.write("\n")
        f.write("Steering score: " + str(steering_scores))
        f.write("\n")
        f.write("mean: " + str( float(sum(steering_scores))/len(steering_scores) ))

        f.write("\n")
        f.write("Speed score: " + str(speed_scores))
        f.write("\n")
        f.write("mean: " + str( float(sum(speed_scores))/len(speed_scores) ))

        f.write("\n")
        f.write("Speed below score: " + str(speed_below_scores))
        f.write("\n")
        f.write("mean: " + str( float(sum(speed_below_scores))/len(speed_below_scores) ))

        f.write("\n")
        f.write("Speed above score: " + str(speed_above_scores))
        f.write("\n")
        f.write("mean: " + str( float(sum(speed_above_scores))/len(speed_above_scores) ))

        f.write("\n")
        f.write("Count below speed_limit: " + str(count_belows))
        f.write("\n")
        f.write("mean: " + str( float(sum(count_belows))/len(count_belows) ))

        f.write("\n")
        f.write("Count above speed_limit: " + str(count_aboves))
        f.write("\n")
        f.write("mean: " + str( float(sum(count_aboves))/len(count_aboves) ))

        f.write("\n")
        f.write("Crossed broke line: " + str(crossed_broke_list))
        f.write("\n")
        f.write("mean: " + str( float(sum(crossed_broke_list))/len(crossed_broke_list) ))

        f.write("\n")
        f.write('Crossed "none" line: ' + str(crossed_none_list))
        f.write("\n")
        f.write("mean: " + str( float(sum(crossed_none_list))/len(crossed_none_list) ))

        f.write("\n")
        f.write("crossed other lines: " + str(crossed_others_list))
        f.write("\n")
        f.write("mean: " + str( float(sum(crossed_others_list))/len(crossed_others_list) ))

        f.write("\n")
        f.write("target_reached_count: " + str(targets_reached))
        f.write("\n")
        f.write("collision_count: " + str(collisions))
        f.write("\n")
        f.write("not_moving_count: " + str(not_moving))
        f.write("\n")
        f.write("\n")
        f.close()


#recording_to_video('../../Training_data')
#for i in range(85, 89):
#    rate_recording(path="/hdd/Test_recordings/Nvidia acvieved 4of4 on t1 and 2of3 on t2/town1", cur_folder=i,file_name="missing-Training/Spatial/Stored_models/36/Checkpoints/val_loss-06-0.102")

#rate_recording(path="../../Test_recordings", cur_folder=6)
#rate_recording(path="../../Test_recordings", cur_folder=4, file_name="3000-Training/Spatial/Stored_models/1/Checkpoints/best-val-08-0.116")
