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
import cv2


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
from Misc.rate_test_results import rate_recording

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
    def __init__(self, carla_world, hud, start_waypoint=-1):
        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.recorder = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.actor_list = []

        self._actor_filter = 'vehicle.bmw.grandtourer'
        self.world.on_tick(hud.on_world_tick)
        self.start_waypoint = start_waypoint
        self.restart()


    def spawn_npc(self, client, safe=False, n_vehicles=30):
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")

        if safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if n_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif n_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, n_vehicles, number_of_spawn_points)
            n_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= n_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        for response in client.apply_batch_sync(batch):
            if response.error:
                logging.error(response.error)
            else:
                self.actor_list.append(response.actor_id)

        print('spawned %d vehicles, press Ctrl+C to exit.' % len(self.actor_list))


    def restart(self):
        # Get a random blueprint.
        print("bluepriont")
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        print("spawn")

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
            spawn_point = self.map.get_spawn_points()[self.start_waypoint]
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
        self.simulation_time = 0
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.simulation_time = timestamp.elapsed_seconds


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
        self.record = False
        self.recording_text = []
        self.images = []
        self.world = world
        self.agent = agent
        self.temp_steering = []
        self.folder_name = folder_name
        self.folder = None
        self.path = path
        self.throttle = 0
        self.brake = 0
        self.controller_updates = 0
        self.camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self._sensor = ['sensor.camera.rgb', cc.Raw, 'Camera RGB']
        server_world = world.player.get_world()
        bp_library = server_world.get_blueprint_library()
        bp = bp_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '460')
        bp.set_attribute('image_size_y', '345')
        #bp.set_attribute('sensor_tick', '0.05')

        self._sensor.append(bp)
        self.sensor = server_world.spawn_actor(
            self._sensor[-1],
            self.camera_transform,
            attach_to=self.world.player)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: Recorder.update(weak_self, image))

    @staticmethod
    def update(weak_self, image):
        self = weak_self()
 
       # Store image beforehand since it will be used by the controller
        self.record_image(image)

        control = self.agent._vehicle.get_control()
        # Store information
        self.record_output(control, image.frame_number)

    def stop_recording(self, stop_condition):
        if len(self.recording_text) > 0:
            # define the name of the directory to be created
            if self.folder_name:
                self.folder = self.folder_name
            else:
                last_folder = 0
                for folder in os.listdir(self.path):
                    try:
                        int(folder)
                    except ValueError as ve:
                        print(ve)
                        continue
                    if int(folder) >= last_folder:
                        last_folder = int(folder)+1
                self.folder = last_folder

            self.path = self.path + "/" + str(self.folder)

            try:
                os.mkdir(self.path)
                os.mkdir(self.path + "/Measurments")
                os.mkdir(self.path + "/Images")
                print("Successfully created the directory %s " % self.path)

            except OSError:
                print ("Creation of the directory %s failed" % self.path)
            
            f = open(self.path + "/" + stop_condition +".txt", "w+")
            f.close()
            keys = self.recording_text[0].keys()
            with open(self.path + '/Measurments/recording.csv', 'w') as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.recording_text)
            self.recording_text = []
            i = 0
            l = len(self.images)
            for image in self.images:
                if i % (math.ceil(l/100)) == 0:
                    print("\r Storing image " + str(i) + " of " + str(l), end="")
                if i % 4 == 0:
                    image.save_to_disk(self.path + '/Images/%08d' % image.frame_number)
                
                i += 1
            self.images = []

    def record_output(self, control, frame_number):
        lane_invasion = self.world.lane_invasion_sensor.lane_invasion
        if len(lane_invasion) > 2:
            self.world.lane_invasion_sensor.lane_invasion = ""

        self.recording_text.append({
            'frame': frame_number,
            'Speed': np.round(self.agent.speed/100, 4),
            'Throttle': control.throttle,
            'Steer': control.steer,
            'Brake': control.brake,
            'Gear': control.gear,
            'speed_limit': float(self.agent.speed_limit)/100,
            'at_TL': self.agent.is_at_traffic_light,
            'TL_state': self.agent.light_state,
            'fps': self.world.hud.server_fps,
            'Direction': self.agent.direction,
            'Upcoming_direction': self.agent.upcoming_direction,
            'Lane_Invasion': self.agent.lane_invasion,
            'Collision': self.world.collision_sensor.collision,
            "real_time(s)": pygame.time.get_ticks() / 1000,
            "Simulation_time(s)": datetime.timedelta(seconds=int(self.world.hud.simulation_time)),
            "Controller_updates": self.controller_updates
        })

    def record_image(self, image):
        """ Convert and append image to images"""
        image.convert(cc.Raw)
        self.images.append(image)

def set_weather(world, random_weather):
    train_weathers = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.SoftRainNoon,


        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudySunset,
        carla.WeatherParameters.WetSunset,
        carla.WeatherParameters.SoftRainSunset,
    ]
    train_weather_names = [
        "ClearNoon",
        "CloudyNoon",
        "WetNoon",
        "SoftRainNoon",

        "ClearSunset",
        "CloudySunset",
        "WetSunset",
        "SoftRainSunset"
    ]
    test_weathers = [
        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.HardRainNoon,

        carla.WeatherParameters.WetCloudySunset,
        carla.WeatherParameters.HardRainSunset,
        carla.WeatherParameters.MidRainSunset,
    ]
    if random_weather == 1:
        index = random.randint(0, len(train_weathers)-1)
        weather = train_weathers[index]
        weather_description = train_weather_names[index]
    else:
        weather = train_weathers[1]
        weather_description = train_weather_names[1]

    world.world.set_weather(weather)
    return weather_description
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
        world = World(client.get_world(), hud, start_waypoint=args.waypoints[0])
        
        #Spawn cars
        if args.cars == 1:
            n_vehicles = 80
            world.spawn_npc(client, n_vehicles=n_vehicles)
        
        #Set weather
        weather_description = set_weather(world, args.random_weather)

        # Ignore traffic light or not
        if args.traffic_light == 0:
            ignore_traffic_light = True
        else:
            ignore_traffic_light = False

        #Spawn agent
        agent = BasicAgent(world.player, autonomous=True, model_path=args.model, model_type=args.model_type, ignore_traffic_light=ignore_traffic_light)

        # Set start positiong and destination
        start_waypoint = world.world.get_map().get_waypoint(agent._vehicle.get_location())
        destination = world.map.get_spawn_points()[args.waypoints[1]]
        agent.set_destination((destination.location.x,
                               destination.location.y,
                               destination.location.z))

        #Get initial distance
        distance = start_waypoint.transform.location.distance(destination.location)
        print("Initial distance: " + str(distance))

        # Initiate the recorder
        recorder = Recorder(world, agent, args.path)
        world.recorder = recorder
        agent._local_planner._vehicle_controller.recorder = recorder
        recorder.record = True
        counter = 0
        fps_que =[]
        stop = False
        not_moving_count = 0
        previous_distance = 0
        while True:
            # as soon as the server is ready continue!
            if not world.world.wait_for_tick(10.0):
                continue

            # Update the agent with information from the world
            agent.update_information(world)

            # Get controller command
            control, actual_steering = agent.run_step(recorder)
            brake = control.brake
            #if control.brake > 0.5:
            #    brake = 1
            #else:
            #    brake = 0
            if control.throttle > brake:
                brake = 0.0
            control.brake = brake
            control.manual_gear_shift = False

            #Apply the control
            world.player.apply_control(control)
            recorder.controller_updates += 1

            if world.collision_sensor.collision:
                print("Stopping recording session due to collision...")
                import time
                time.sleep(1)
                stop_condition = "collision"
                stop = True

            # Stop recorder when target destination has been reached
            if len(agent._local_planner._waypoints_queue) < 1:
                print("Target Reached, stopping recording session...")
                stop_condition = "target_reached"
                stop = True

            counter += 1
            fps_que.append(world.hud.server_fps)

            if counter % 200 == 0:
                print("step: " + str(counter))
                cur_waypoint = world.world.get_map().get_waypoint(agent._vehicle.get_location())
                distance = cur_waypoint.transform.location.distance(destination.location)
                print("Distance to goal= " + str(distance))
                if abs(distance - previous_distance) < 2:
                    not_moving_count += 1
                else:
                    not_moving_count = 0
                previous_distance = distance
                if not_moving_count > 20:
                    print("Not moving anymore... quiting recording")
                    stop_condition = "not_moving"
                    stop = True

            if stop:
                recorder.record = False
                if recorder.sensor is not None:
                    recorder.sensor.destroy()
                if world.collision_sensor.sensor is not None:
                    world.collision_sensor.sensor.destroy()
                if world.lane_invasion_sensor.sensor is not None:
                    world.lane_invasion_sensor.sensor.destroy()
                if world is not None:
                    world.player.destroy()
                    print('\ndestroying %d actors' % len(world.actor_list))
                    client.apply_batch([carla.command.DestroyActor(x) for x in world.actor_list])

                if counter < 50:
                    print("Didn't get far enough, not storing recording")
                else:
                    print("Storing images and measurments...")
                    recorder.stop_recording(stop_condition)
                return



    #except IOError as (errno, strerror):
    #    print("Error in main loop: error({0}): {1}".format(errno, strerror))
    except TypeError as t:
        print("Error in main loop.")
        print(t)
    except NameError as n:
        print("Error in main loop.")
        print(n)
    except AttributeError as a:
        print("Error in main loop.")
        print(a)
    except SyntaxError as s:
        print("Error in main loop.")
        print(s)
        exit()
    except IndexError as i:
        print("Error in main loop.")
        print(i)
        exit()
    except ValueError as v:
        print("Error in main loop.")
        print(v)
        exit()
    except:
        print("Unexpected error:", sys.exc_info()[0])
        exit()
    
    finally:
        print("EXITING")
        if recorder.folder:
            if args.model == None:
                args.model = "./Training/Temporal/Current_model/" + os.listdir("./Training/Spatial/Current_model/")[0]
                file_name = str(counter)+"-" + args.model.split(".")[0] + "." + args.model.split(".")[1]
            else:
                file_name = str(counter)+"-" + args.model.split(".")[0] + "." +  args.model.split(".")[1]
            recording_to_video(path=args.path, cur_folder=recorder.folder, file_name=file_name)
            rate_recording(path=args.path, cur_folder=recorder.folder, file_name=file_name)
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
        '-w', '--random_weather',
        metavar='W',
        default=1,
        type=int,
        help='set to 0 use clear noon every time')
    argparser.add_argument(
        '-c', '--cars',
        metavar='W',
        default=1,
        type=int,
        help='set to 0 to not include cars')
    argparser.add_argument(
        '-t', '--traffic_light',
        metavar='W',
        default=1,
        type=int,
        help='set to 0 to ignore traffic lights')
    argparser.add_argument(
        '--model',
        default=None,
        help='Where to store data')
    argparser.add_argument(
        '--model_type',
        default="Spatiotemporal",
        help='Which modeltype to use (spatial, temporal, spatiotemporal)')
    argparser.add_argument(
        '--waypoints',
        default=[-1, 0], #[-1, 0],
        help='Waypoint index of where to start and stop')
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
