#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
    Example of automatic vehicle control from client side.
"""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref
import csv

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('carla')[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
from agents.navigation.basic_agent import *
from agents.navigation.local_planner import RoadOption
from Misc.recording_to_video import recording_to_video

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, hud):
        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.recorder = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None

        self._actor_filter = 'vehicle.bmw.grandtourer'
        self.world.on_tick(hud.on_world_tick)
        self.restart()

    def restart(self):
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.player.destroy()
            if self.collision_sensor is not None:
                self.collision_sensor.sensor.destroy()
            if self.lane_invasion_sensor is not None:
                self.lane_invasion_sensor.sensor.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_point = self.map.get_spawn_points()[-1]
            #spawn_points = self.map.get_spawn_points()
            #spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        self.collision_sensor = CollisionSensor(self.player)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player)


# ==============================================================================
# -- HUD -----------------------------------------------------------
# ==============================================================================

class HUD(object):
    def __init__(self):
        self.server_fps = 0
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.collision = False
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        print('Collision with %r' % actor_type)
        print("with intensity: "+ str(intensity))
        self.collision = True



# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor):
        self.lane_invasion = ""
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]

        self.lane_invasion = 'Crossed line %s' % ' and '.join(text)
        print(self.lane_invasion)

# ==============================================================================
# -- Recorder() ---------------------------------------------------------------
# ==============================================================================

class Recorder():
    def __init__(self, world, agent, path, folder_name=None):
        self.recording_text = []
        self.images = []
        self.world = world
        self.agent = agent
        self.temp_steering = []
        self.folder_name = folder_name
        self.path = path
        self.camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self._sensor = ['sensor.camera.rgb', cc.Raw, 'Camera RGB']
        server_world = world.player.get_world()
        bp_library = server_world.get_blueprint_library()
        bp = bp_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '320')
        bp.set_attribute('image_size_y', '240')
        #bp.set_attribute('fov', '120')
        bp.set_attribute('sensor_tick', '0.20')

        self._sensor.append(bp)
        self.sensor = server_world.spawn_actor(
            self._sensor[-1],
            self.camera_transform,
            attach_to=self.world.player)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: Recorder.record(weak_self, image))

    @staticmethod
    def record(weak_self, image):
        self = weak_self()
        #if self.agent._local_planner._target_road_option != None:
        self.record_output(self.world, image.frame_number)
        self.record_image(image)


    def stop_recording(self):
        print(len(self.recording_text))
        if len(self.recording_text) > 0:
            # define the name of the directory to be created
            if self.folder_name:
                folder = self.folder_name
            else:
                last_folder = 0
                for folder in os.listdir(self.path):
                    if folder == ".DS_Store" or folder == "store.h5":
                        continue
                    if int(folder) >= last_folder:
                        last_folder = int(folder)+1
                folder = last_folder

            self.path = self.path + "/" + str(folder)
            print(self.path)

            try:
                os.mkdir(self.path)
                os.mkdir(self.path + "/Measurments")
                os.mkdir(self.path + "/Images")

            except OSError:
                print ("Creation of the directory %s failed" % self.path)
            else:
                print ("Successfully created the directory %s " % self.path)
            print(len(self.images))
            keys = self.recording_text[0].keys()
            with open(self.path + '/Measurments/recording.csv', 'wb') as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.recording_text)
            self.recording_text = []
            for image in self.images:
                image.save_to_disk(self.path + '/Images/%08d' % image.frame_number)
            self.images = []

    def record_output(self, world, frame_number):
        lane_invasion = world.lane_invasion_sensor.lane_invasion
        if len(lane_invasion) > 2:
            world.lane_invasion_sensor.lane_invasion = ""
        control = world.player.get_control()
        v = world.player.get_velocity()
        speed_limit = world.player.get_speed_limit()
        is_at_traffic_light = world.player.is_at_traffic_light()
        traffic_light = world.player.get_traffic_light()
        traffic_light_state = world.player.get_traffic_light_state()
        self.recording_text.append({
            'frame': frame_number,
            'Speed': np.round((3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))/100, 4),
            'Throttle': control.throttle,
            'Steer': control.steer,
            'Brake': control.brake,
            'Reverse': control.reverse,
            'Hand brake': control.hand_brake,
            'Manual': control.manual_gear_shift,
            'Gear': control.gear,
            'speed_limit': float(speed_limit)/100,
            'at_TL': is_at_traffic_light,
            'TL': traffic_light,
            'TL_state': traffic_light_state,
            'fps': self.world.hud.server_fps,
            'Direction': self.agent._local_planner._target_road_option,
            'Lane_Invasion': lane_invasion,
            'Collision': self.world.collision_sensor.collision
        })

    def record_image(self, image):
        image.convert(cc.Raw)
        self.images.append(image)

# ==============================================================================
# -- game_loop() ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        hud = HUD()
        world = World(client.get_world(), hud)

        agent = BasicAgent(world.player, autonomous=True)

        #start_waypoint = world.world.get_map().get_waypoint(agent._vehicle.get_location())
        start_waypoint = world.world.get_map().get_waypoint(agent._vehicle.get_location())
        #destination = random.choice(world.map.get_spawn_points())
        destination = world.map.get_spawn_points()[0]
        agent.set_destination((destination.location.x,
                               destination.location.y,
                               destination.location.z))

        distance = start_waypoint.transform.location.distance(
                destination.location)

        print("Initial distance: " + str(distance))
        recorder = Recorder(world, agent, args.path)
        world.recorder = recorder
        counter = 0
        fps_que =[]
        distance_que = []
        stop = False

        while True:
            # as soon as the server is ready continue!
            if not world.world.wait_for_tick(10.0):
                continue

            if world.collision_sensor.collision:
                print("Stopping recording session due to collision...")
                import time
                time.sleep(1)
                stop = True

            # Stop recorder when target destination has been reached
            if len(agent._local_planner._waypoints_queue) == 0:
                print("Target Reached, stopping recording session...")
                stop = True

            counter += 1
            fps_que.append(world.hud.server_fps)

            if counter % 100 == 0:
                print("step: " + str(counter))
                print("average fps: " + str(sum(fps_que)/len(fps_que)))
                fps_que = []
                cur_waypoint = world.world.get_map().get_waypoint(agent._vehicle.get_location())
                distance = cur_waypoint.transform.location.distance(destination.location)
                distance_que.append(math.ceil(distance))
                print("Distance to goal: " + str(distance))
                if len(distance_que) > 10:
                    distance_que = distance_que[-10:]
                    if len(set(distance_que)) == 1:
                        print("Not moving anymore... quiting recording")
                        stop = True

            if stop:
                if recorder.sensor is not None:
                    recorder.sensor.destroy()
                if world.collision_sensor.sensor is not None:
                    world.collision_sensor.sensor.destroy()
                if world.lane_invasion_sensor.sensor is not None:
                    world.lane_invasion_sensor.sensor.destroy()
                if world is not None:
                    world.player.destroy()
                print("Storing images and measurments...")
                recorder.stop_recording()
                return


            speed_limit = world.player.get_speed_limit()
            agent._local_planner.set_speed(speed_limit)
            control = agent.run_step(recorder)
            if control.throttle < 0.5:
                control.throttle = 0
            else:
                control.throttle = 1

            if control.brake >0.5:
                control.brake = 1
            else:
                control.brake = 0
            
            world.player.apply_control(control)

    finally:
        recording_to_video(path="Test_recordings")
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '--path',
        default='Test_recordings',
        help='Where to store data')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument("-a", "--agent", type=str,
                           choices=["Roaming", "Basic"],
                           help="select which agent to run",
                           default="Basic")
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
