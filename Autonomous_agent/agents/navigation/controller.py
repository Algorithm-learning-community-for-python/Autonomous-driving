#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math
import numpy as np
from keras.models import load_model
import h5py
import carla
from agents.tools.misc import get_speed

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from enum import Enum

from Spatial.data_configuration import Config
from Misc.preprocessing import get_one_hot_encoded

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
tf_config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=tf_config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

class ImitatorController():
    def __init__(self, vehicle, local_planner):
        self.temporal = False
        self.direction_sequence = []
        self.image_sequence = []
        self.conf = Config()
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.model = None
        if self. temporal:
            self.model = load_model("./Training/Temporal/Current_model/model.h5")
        else:
            self.model = load_model("./Training/Spatial/Current_model/best_val.hdf5")

        self.local_planner = local_planner
        self.direction_categories = [
            "RoadOption.VOID", 
            "RoadOption.LEFT",
            "RoadOption.RIGHT",
            "RoadOption.STRAIGHT",
            "RoadOption.LANEFOLLOW",
            "RoadOption.CHANGELANELEFT",
            "RoadOption.CHANGELANERIGHT"
        ]
        self.img_size = self.conf.input_size_data["Image"]


    def run_step(self,recorder):
        if self.temporal:
            return self.run_temporal_step(recorder)
        else:
            return self.run_spatial_step(recorder)

    def get_direction(self):
        #print("direction")

        direction = self.local_planner._target_road_option
        direction = [str(direction)]
        direction = get_one_hot_encoded(self.conf.direction_categories, direction)
        return np.array(direction).reshape(1,7)

    def get_tl_state(self):
        #print("tl_state")
        tl_state = self._vehicle.get_traffic_light_state()
        tl_state = [str(tl_state)]
        tl_state = get_one_hot_encoded(self.conf.tl_categories, tl_state)
        return np.array(tl_state).reshape(1,3)

    def get_speed(self):
        #print("speed")

        v = self._vehicle.get_velocity()
        speed = (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))/100
        return np.array([speed])

    def get_speed_limit(self):
        #print("speed_limit")

        speed_limit = self._vehicle.get_speed_limit()
        return np.array([float(speed_limit)/100])

    def get_image(self, recorder):
        #print("IMAGE")
        img = self.to_rgb_array(recorder.images[-1])
        img = img[self.conf.top_crop:, :, :]
        return np.array(img).reshape(1, self.img_size[0], self.img_size[1], self.img_size[2])

    def run_spatial_step(self, recorder):
        """ Predicts and returns output """
        if recorder.images:
            X = {
                "input_Image": self.get_image(recorder),
                "input_Direction": self.get_direction(),
                "input_Speed": self.get_speed(),
                "input_speed_limit": self.get_speed_limit(),
                "input_TL_state": self.get_tl_state()
            }
            #print("################     PREDICTING      ################")
            control = self.model.predict(X, batch_size=1)
            control = {
                out.name.split(':')[0].encode("ascii").split("/")[0]: control[i][0] for i, out in enumerate(self.model.outputs)
            }
        else:
            #Default value if no prediction
            control = {"output_Throttle":[0], "output_Brake":[0], "output_Steer": [0]}

        #print("################     Setting controls      ################")
        throttle = float(control["output_Throttle"][0])
        brake = float(control["output_Brake"][0])
        steering = float(control["output_Steer"][0])

        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = False
        control.manual_gear_shift = False

        return control
    
    def run_temporal_step(self, recorder):
        current_speed = get_speed(self._vehicle)
        direction = self.get_direction()

        self.direction_sequence.append(direction[0])
        #print(self.direction_sequence)
        if len(self.direction_sequence) > self.conf.input_size_data["Sequence_length"]:
            self.direction_sequence = self.direction_sequence[1:]

        if recorder.images:
            img = self.to_rgb_array(recorder.images[-1])
            img = img[self.conf.top_crop:, :,:]
            #print("################     PREDICTING      ################")
            #TODO: RUN NN to predict throttle and steering
            self.image_sequence.append(np.array(img))
            if len(self.image_sequence) > self.conf.input_size_data["Sequence_length"]:
                self.image_sequence = self.image_sequence[1:]
                image_size = [1] + [self.conf.input_size_data["Sequence_length"]] + self.conf.input_size_data["Image"]
                direction_size = [1] + [self.conf.input_size_data["Sequence_length"]] + self.conf.input_size_data["Direction"]
                control = self.model.predict([np.array(self.image_sequence).reshape(image_size), np.array(self.direction_sequence).reshape(direction_size)], batch_size=1)
            else:
                control = [[0]]
            #print(control)
        else:
            control = [[0]]
        #print(recorder.speed)
        if recorder.speed > 20:
            throttle = 0
        else:
            throttle = 1 

        steering = float(control[0][0]) #self.get_steering()

        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

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
