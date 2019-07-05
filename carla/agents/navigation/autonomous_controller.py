#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#pylint: disable=superfluous-parens

""" This module contains PID controllers to perform lateral and longitudinal control. """
import os
import math
import sys
import cv2
from collections import deque
from enum import Enum
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from agents.tools.misc import distance_vehicle, draw_waypoints

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from Misc.preprocessing import get_one_hot_encoded

import carla

TF_CONF = tf.ConfigProto()
TF_CONF.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# tf_config.log_device_placement = True # to log device placement(on which device the operation ran)
# (nothing gets printed in Jupyter, only if you run it standalone)
SESSION = tf.Session(config=TF_CONF)
set_session(SESSION)  # set this TensorFlow session as the default session for Keras

class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations 
    when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

class ImitatorController(object):
    def __init__(self, vehicle, local_planner, agent, model_path=None, model_type=None):

        self.temporal = False
        self.spatial = False
        self.spatiotemporal = False
        self.direction_sequence = []
        self.speed_sequence = []
        self.speed_limit_sequence = []
        self.tl_state_sequence = []
        self.image_sequence = []
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.model = None
        self.light_state = None
        self.local_planner = local_planner
        self.agent = agent
        self.recorder = None
        self.previous_directions = []
        print("Clearing session...")
        K.clear_session()

        print("Loading config...")
        if model_type == "Spatiotemporal":
            self.spatiotemporal = True
            from Spatiotemporal.data_configuration import Config
        elif model_type == "Temporal":
            self.temporal = True
            from Temporal.data_configuration import Config
        else:
            self.spatial = True
            from Spatial.data_configuration import Config
        self.conf = Config()
        self.img_size = self.conf.input_size_data["Image"]

        print("Loading model:")
        print("Model type: " + str(model_type))
        print("Model path: " + str(model_path))
        if not model_path:
            if model_type:
                cur_model_path = "./Training/" + model_type + "/Current_model/"
                self.model = load_model(cur_model_path + os.listdir(cur_model_path)[0])
            else: 
                print("Model type not specified, exiting...")
                exit()
        else:
            print(model_path)
            self.model = load_model(model_path)
        global graph
        graph = tf.get_default_graph() 


        seq_len = self.conf.input_size_data["Sequence_length"]
        self.X = {
            "input_Image": np.zeros([1, seq_len] + self.conf.input_size_data["Image"]),
            "input_Direction": np.zeros([1, seq_len] + self.conf.input_size_data["Direction"]),
            "input_Speed": np.zeros([1, seq_len] + self.conf.input_size_data["Speed"]),
            "input_ohe_speed_limit": np.zeros([1, seq_len] + self.conf.input_size_data["ohe_speed_limit"]),
            "input_TL_state": np.zeros([1, seq_len] + self.conf.input_size_data["TL_state"]) 
            }
        self.direction_categories = [
            "RoadOption.VOID", 
            "RoadOption.LEFT",
            "RoadOption.RIGHT",
            "RoadOption.STRAIGHT",
            "RoadOption.LANEFOLLOW",
            "RoadOption.CHANGELANELEFT",
            "RoadOption.CHANGELANERIGHT"
        ]
        print("INIT DONE")


    def run_step(self):
        if self.temporal:
            return self.run_temporal_step()
        elif self.spatiotemporal:
            return self.run_spatiotemporal_step()
        else:
            return self.run_spatial_step()

    def get_direction(self):
        direction = self.agent.direction
        #self.previous_directions.append(direction)
        #if direction.value == RoadOption.LANEFOLLOW.value:
        #    _, upcoming_direction = self.agent._local_planner.get_upcoming_waypoint_and_direction(2)
        #    if upcoming_direction != None and upcoming_direction.value != RoadOption.LANEFOLLOW.value:
        #       direction = upcoming_direction
        
        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self.local_planner._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self.local_planner._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self.local_planner._waypoint_buffer.popleft()
        
        if direction.value == RoadOption.LANEFOLLOW.value:
            upcoming_direction = None
            upcoming_wp = None
            l = len(self.local_planner._waypoint_buffer)

            # Look in the current buffer after intersections
            if l >= 1:
                for (wp, updir) in self.local_planner._waypoint_buffer:
                    if updir != None and updir.value != RoadOption.LANEFOLLOW.value:
                        upcoming_direction = updir
                        upcoming_wp = wp
                        break

            # If no intersection was found in the buffer, then look in the waypoints que
            if upcoming_direction is None:
                peek_distance = 5 - l
                l2 = len(self.local_planner._waypoints_queue)
                if l2 < peek_distance:
                    peek_distance = l2

                for i in range(peek_distance):
                    wp, updir = self.agent._local_planner.get_upcoming_waypoint_and_direction(i)
                    if updir != None and updir.value != RoadOption.LANEFOLLOW.value:
                        upcoming_direction = updir
                        upcoming_wp = wp
                        break
            
            if upcoming_direction is not None:
                direction = upcoming_direction
                self.agent.upcoming_direction = upcoming_direction
            else:
                self.agent.upcoming_direction = RoadOption.LANEFOLLOW
            #draw_waypoints(self._vehicle.get_world(), [upcoming_wp], self._vehicle.get_location().z + 1.0)
        else:
            self.agent.upcoming_direction = RoadOption.LANEFOLLOW

        direction = [str(direction)]
        direction = get_one_hot_encoded(self.conf.direction_categories, direction)

        return np.array(direction).reshape(1, self.conf.input_size_data["Direction"][0])

    def get_tl_state(self):
        tl_state = get_one_hot_encoded(self.conf.tl_categories, [self.agent.light_state])
        return np.array(tl_state).reshape(1, self.conf.input_size_data["TL_state"][0])

    def get_speed(self):
        return np.array([np.round(self.agent.speed/100, 4)])

    def get_speed_limit(self):
        speed_limit = self.agent.speed_limit
        if self.conf.input_data["ohe_speed_limit"]:
            speed_limit = [float(speed_limit/100)]
            speed_limit = get_one_hot_encoded(self.conf.sl_categories, speed_limit)
            return np.array(speed_limit).reshape(1, self.conf.input_size_data["ohe_speed_limit"][0])
        else:
            return np.array([float(speed_limit)/100])

    def get_image(self, i=-1):
        img = self.to_rgb_array(self.recorder.images[i])
        img = img[self.conf.top_crop:, :, :]
        img = cv2.resize(img,(self.img_size[1], self.img_size[0]))
        return np.array(img).reshape(1, self.img_size[0], self.img_size[1], self.img_size[2])

    def run_spatial_step(self):
        """ Predicts and returns output """
        #print("SETTING INPUT")
        if self.conf.input_data["ohe_speed_limit"]:
            sl = "input_ohe_speed_limit"
        else:
            sl = "input_speed_limit"
        if self.recorder.images:
            #print("Setting input")
            X = {
                "input_Image": self.get_image(),
                "input_Direction": self.get_direction(),
                "input_Speed": self.get_speed(),
                sl: self.get_speed_limit(),
                "input_TL_state": self.get_tl_state()
            }
            #print("################     PREDICTING      ################")
            with graph.as_default():
                control = self.model.predict(X, batch_size=1)
            control =  {
                out.name.split(':')[0].split("/")[0]: control[i][0] for i, out in enumerate(self.model.outputs)
            }
        else:
            #Default value if no prediction
            control = {"output_Throttle":[0], "output_Brake":[0], "output_Steer": [0]}

        #print("################     Setting controls      ################")
        #print(control)
        throttle = float(control["output_Throttle"][0])
        brake = float(control["output_Brake"][0])
        steering = float(control["output_Steer"][0])

        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = False
        control.manual_gear_shift = False
        #print("Returning control")
        #print(control)
        return control

    def run_spatiotemporal_step(self):
        seq_len = self.conf.input_size_data["Sequence_length"]
        step_size = self.conf.step_size_testing
        if self.conf.input_data["ohe_speed_limit"]:
            speed_limit_rep = "input_ohe_speed_limit"
        else:
            speed_limit_rep = "input_speed_limit"
        
        current_speed = self.get_speed()
        self.speed_sequence.append(current_speed)
        if len(self.speed_sequence) > seq_len*step_size: 
            self.speed_sequence = self.speed_sequence[1:]

        direction = self.get_direction()
        self.direction_sequence.append(direction)
        if len(self.direction_sequence) > seq_len*step_size: 
            self.direction_sequence = self.direction_sequence[1:]

        tl_state = self.get_tl_state()
        self.tl_state_sequence.append(tl_state)
        if len(self.tl_state_sequence) > seq_len*step_size: 
            self.tl_state_sequence = self.tl_state_sequence[1:]
        
        speed_limit = self.get_speed_limit()
        self.speed_limit_sequence.append(speed_limit)
        if len(self.speed_limit_sequence) > seq_len*step_size: 
            self.speed_limit_sequence = self.speed_limit_sequence[1:]

        if len(self.recorder.images) > seq_len*step_size and len(self.speed_limit_sequence) >= seq_len*step_size:   
            #print("################     CREATING INPUT      ################")
            X = {}
            index = 0
            #print(len(self.speed_limit_sequence))
            for i in range(((seq_len-1)*step_size+1), 0, -step_size):
                
                #print(i)
                X["input_Image"+str(index)] = self.get_image(-i)
                X["input_Direction"+str(index)] = self.direction_sequence[-i]
                X["input_Speed"+str(index)] = self.speed_sequence[-i]
                X[speed_limit_rep+str(index)] = self.speed_limit_sequence[-i]
                X["input_TL_state"+str(index)] = self.tl_state_sequence[-i]
                index += 1

            #print("################     PREDICTING      ################")
            #print(X)
            with graph.as_default():

                control = self.model.predict(X, batch_size=1)
            control =  {
                out.name.split(':')[0].split("/")[0]: control[i][0] for i, out in enumerate(self.model.outputs)
            }
        else:
            #Default value if no prediction
            control = {"output_Throttle":[0], "output_Brake":[0], "output_Steer": [0]}

        #print("################     Setting controls      ################")
        #print(control)
        throttle = float(control["output_Throttle"][0])
        brake = float(control["output_Brake"][0])
        steering = float(control["output_Steer"][0])

        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = False
        control.manual_gear_shift = False
        #print("Returning control")
        #print(control)
        return control

    def run_temporal_step(self):
        seq_len = self.conf.input_size_data["Sequence_length"]
        step_size = self.conf.step_size_testing
        
        current_speed = self.get_speed()
        self.speed_sequence.append(current_speed)
        if len(self.speed_sequence) > seq_len*step_size: 
            self.speed_sequence = self.speed_sequence[1:]

        direction = self.get_direction()
        self.direction_sequence.append(direction)
        if len(self.direction_sequence) > seq_len*step_size: 
            self.direction_sequence = self.direction_sequence[1:]

        tl_state = self.get_tl_state()
        self.tl_state_sequence.append(tl_state)
        if len(self.tl_state_sequence) > seq_len*step_size: 
            self.tl_state_sequence = self.tl_state_sequence[1:]
        
        speed_limit = self.get_speed_limit()
        self.speed_limit_sequence.append(speed_limit)
        if len(self.speed_limit_sequence) > seq_len*step_size: 
            self.speed_limit_sequence = self.speed_limit_sequence[1:]

        if len(self.recorder.images) > seq_len*step_size and len(self.speed_limit_sequence) >= seq_len*step_size:   
            #print("################     CREATING INPUT      ################")
            index = 0
            try:
                for i in range(((seq_len-1)*step_size+1), 0, -step_size):
                    self.X["input_Image"][0, index, :, :, :] = self.get_image(-i)
                    self.X["input_Direction"][0, index, :] = self.direction_sequence[-i]
                    self.X["input_Speed"][0, index, :] = self.speed_sequence[-i]
                    self.X["input_ohe_speed_limit"][0, index, :] = self.speed_limit_sequence[-i]
                    self.X["input_TL_state"][0, index, :] = self.tl_state_sequence[-i]
                    index += 1
            except IndexError as i:
                print("IndexError in prediction...")
                print(i)
                exit()
            except:
                print("Unexpected error when setting X:", sys.exc_info()[0])
                raise
            #print("################     PREDICTING      ################")
            #print(self.X)
            try:
                with graph.as_default():
                    control = self.model.predict(self.X, batch_size=1)
            except ValueError as v:
                print("ValueError in prediction...")
                print(v)
                exit()
            except AttributeError as a:
                print("AttributeError in prediction...")
                print(a)
                exit()
            except:
                print("Unexpected error when predicing:", sys.exc_info()[0])
                raise
            control =  {
                out.name.split(':')[0].split("/")[0]: control[i][0] for i, out in enumerate(self.model.outputs)
            }
        else:
            #Default value if no prediction
            control = {"output_Throttle":[0], "output_Brake":[0], "output_Steer": [0]}

        #print("################     Setting controls      ################")
        #print(control)
        throttle = float(control["output_Throttle"][0])
        brake = float(control["output_Brake"][0])
        steering = float(control["output_Steer"][0])

        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = False
        control.manual_gear_shift = False
        #print("Returning control")
        #print(control)
        return control
    
    def to_bgra_array(self, image):
        """Convert a CARLA raw image to a BGRA np array."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        return array

    def to_rgb_array(self, image):
        """Convert a CARLA raw image to a RGB np array."""
        array = self.to_bgra_array(image)
        # Convert BGRA to RGB.
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array
