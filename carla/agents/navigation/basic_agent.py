#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """


import carla
import random
import math
import numpy as np
from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import RoadOption

from agents.tools.misc import get_speed

class BasicAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, autonomous=False, target_speed=20, model_path=None, start_waypoint=None, model_type=None, ignore_traffic_light=False, add_noise=False):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(BasicAgent, self).__init__(vehicle)
        self.ignore_traffic_light = ignore_traffic_light
        self.autonomous = autonomous
        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        args_lateral_dict = {
            'K_P': 0.9,
            'K_D': 0.003,
            'K_I': 0,
            'dt': 1.0/20.0}
        args_longitudinal_dict = {
            'K_P': 0.1,
            'K_D': 0.0,
            'K_I': 0,
            'dt': 1.0/20.0}
        self._local_planner = LocalPlanner(
            self, self._vehicle, opt_dict={'target_speed' : target_speed,
            'lateral_control_dict': args_lateral_dict, "longitudinal_control_dict": args_longitudinal_dict}, autonomous=autonomous, model_path=model_path, model_type=model_type)
        self._hop_resolution = 3.0
        self._path_seperation_hop = 2
        self._path_seperation_threshold = 0.5
        self._grp = None
        self.look_ahead_steps = 4

        # Noise calculation
        self.noise_count = 0
        self.turn_right = 0
        self.noise_pause = 0
        self.add_noise = add_noise

        # Vehicle information
        self.speed = 0
        self.speed_limit = 0
        self.lane_invasion = None
        self.is_at_traffic_light = 0
        self.light_state = "Green"
        self.direction = None
        self.upcoming_direction = None
        self.upcoming_waypoint = None
        self.upcoming_direction_short = None

    def update_information(self, world):
        self.speed = get_speed(self._vehicle)
        self.speed_limit = world.player.get_speed_limit()
        self._local_planner.set_speed(self.speed_limit)
        self.direction = self._local_planner._target_road_option
        if self.direction is None:
            self.direction = RoadOption.LANEFOLLOW
        # During autonomous mode, this will be set by the controller
        if not self.autonomous:
            self.upcoming_waypoint, self.upcoming_direction = self._local_planner.get_upcoming_waypoint_and_direction(steps=self.look_ahead_steps)
            if self.upcoming_direction is None:
                self.upcoming_direction = RoadOption.LANEFOLLOW
            
            _, self.upcoming_direction_short = self._local_planner.get_upcoming_waypoint_and_direction(2)
            
            if self.upcoming_direction_short is None:
                self.upcoming_direction_short = RoadOption.LANEFOLLOW

        self.is_at_traffic_light = world.player.is_at_traffic_light()
        if self.ignore_traffic_light:
            self.light_state = "Green"
        else:
            self.light_state = str(self._vehicle.get_traffic_light_state())

    def set_destination(self, location):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """

        start_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        end_waypoint = self._map.get_waypoint(
            carla.Location(location[0], location[1], location[2]))

        route_trace = self._trace_route(start_waypoint, end_waypoint)
        assert route_trace

        self._local_planner.set_global_plan(route_trace)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """
        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), sampling_resolution=self._hop_resolution, world=self._vehicle.get_world())
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def run_step(self, recorder, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """
        self._state = AgentState.NAVIGATING
        actual_steer = None

        if self.autonomous:
            control = self._local_planner.run_step(debug=debug)
            return control, actual_steer # Todo: add steer from pid controller to get loss during driving

        # USE PID CONTROLLERE
        else:
            # Brake immediatly if a red light is seen
            if self.light_state == "Red":
                recorder.adding_noise = False
                return self.emergency_stop(), actual_steer
            
            #Slow down immediatly if the car is driving above the speed limit
            if self.speed - self.speed_limit > 5:
                control = self._local_planner.run_step(debug=debug)
                control.brake = 1
                control.throttle = 0
                return control, actual_steer

            # Check if there is a vehicle in front, and slow down appropriately
            vehicle_list = self._world.get_actors().filter("*vehicle*")
            vehicle_state, vehicle, distance = self._is_vehicle_hazard(vehicle_list)
            if vehicle_state:
                recorder.adding_noise = False
                # Emergency brake if the car is within 10 meters
                if distance == 10:
                    self._state = AgentState.BLOCKED_BY_VEHICLE
                    return self.emergency_stop(), actual_steer

                self._state = AgentState.NAVIGATING
                vehicle_speed = get_speed(vehicle)
                # Drive slowly up to the car in front if it is standing still
                if vehicle_speed < 1 and self.speed < 10:
                    control = self._local_planner.run_step(target_speed=5, debug=debug)
                # Follow the speed of the car in front
                else:
                    control = self._local_planner.run_step(target_speed=vehicle_speed, debug=debug)

                # If speed is high compared to the car in front, then apply brakes
                if distance == 15 and self.speed - vehicle_speed > 10:
                    control.brake = 1
                    control.throttle = 0
                elif distance == 20 and self.speed - vehicle_speed > 20:
                    control.brake = 1
                    control.throttle = 0
            else:
                self._state = AgentState.NAVIGATING
                # Drive slow if the vehicle is in a intersection
                if self.direction == RoadOption.RIGHT or \
                    self.direction == RoadOption.LEFT:
                    control = self._local_planner.run_step(target_speed=20, debug=debug)

                # Slow down if the vehicle is approaching an intersection
                #elif self.upcoming_direction == RoadOption.LEFT or \
                #    self.upcoming_direction == RoadOption.RIGHT:
                #    control = self._local_planner.run_step(target_speed=20, debug=debug)

                # Detects sharp turn and slows down
                elif self.sharp_turn():
                    control = self._local_planner.run_step(target_speed=20, debug=debug)


                # Calculate controller based on no turn, traffic light or vehicle in fron
                else:
                    control = self._local_planner.run_step(debug=debug)
                
                # Add noise to the steering command
                if self.add_noise:
                    self.noise_pause += 1
                    r = random.randint(0, 20)
                    if r == 0 and not recorder.adding_noise and \
                            self.noise_pause > 20 and \
                            abs(control.steer) < 0.05 and \
                            (self.direction == RoadOption.LANEFOLLOW or self.direction == RoadOption.VOID)\
                            and self.speed > 2 and \
                            self.speed < 50:
                        self.turn_right = random.randint(0, 1)
                        self.noise_count = 1
                        recorder.adding_noise = True
                        print("ADDING NOISE")

                    if recorder.adding_noise:
                        if self.speed < 35:
                            momentum = 0.025
                            steps = 8
                        else:
                            momentum = 0.01
                            steps = 4

                        actual_steer = control.steer
                        if self.turn_right == 1:
                            control.steer = self.noise_count*momentum
                        else:
                            control.steer = self.noise_count*-momentum

                        self.noise_count += 1
                        if self.noise_count > steps:
                            recorder.adding_noise = False
                            self.noise_pause = 0
                            print("ADDING NOISE STOPPED")
        return control, actual_steer

    def sharp_turn(self):
        turns = [
            [0,11],
            [11,8],
            [8,14],
            [3,13],
            [13,15],
            [15,20],
            [20,5],
            #opposite way
            [8,11],
            [11,0],
            [7,14],
            [14,8],
            [15,13],
            [13,3],
            [5,20],
            [20,15]
        ]
        
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        if self.upcoming_waypoint.road_id != ego_vehicle_waypoint.road_id and \
                self.upcoming_waypoint.lane_id != ego_vehicle_waypoint.lane_id and \
                self.upcoming_direction == RoadOption.LANEFOLLOW and self.direction == RoadOption.LANEFOLLOW and \
                [ego_vehicle_waypoint.road_id, self.upcoming_waypoint.road_id] in turns:
            return True
        else:
            return False