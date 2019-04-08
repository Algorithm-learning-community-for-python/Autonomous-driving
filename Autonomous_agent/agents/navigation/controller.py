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

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

class ImitatorController():
    def __init__(self, vehicle):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.model = load_model("./Training/model.h5")


    def run_step(self,recorder):
        current_speed = get_speed(self._vehicle)

        if recorder.images:
            img = self.to_rgb_array(recorder.images[-1])

            #TODO: RUN NN to predict throttle and steering
            control = self.model.predict(np.array(img).reshape(1,90,160,3), batch_size=1)
        else:
            control = [[0]]
        print(recorder.speed)
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