"""
🚦 AI-Powered Traffic Management System
📋 Smart Adaptive Traffic Light Control with Priority System
🏛️ Government Institution Integration: Hospitals, Fire, Police, Civil Defense

Author: AI Assistant
Version: 6.2 (Fast Emergency + Smart Detection + Congestion Priority)
Description: Faster emergency vehicles, smart detection near intersection only,
             congestion-based priority system.
"""

import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pandas as pd
from datetime import datetime
import sqlite3
import json
import os
import math
import threading
import time
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

# ============================================
# CONFIGURATION & CONSTANTS
# ============================================

class Config:
    SCREEN_WIDTH = 1400
    SCREEN_HEIGHT = 800
    FPS = 40

    ROAD_WIDTH = 100
    LANE_WIDTH = 35

    MAX_VEHICLES = 25
    SPAWN_INTERVAL = 25
    NORMAL_SPEED_MIN = 1.5
    NORMAL_SPEED_MAX = 2.0
    EMERGENCY_SPEED = 2.0  # FASTER emergency vehicles
    EMERGENCY_ACCELERATION = 0.1  # Faster acceleration
    ACCELERATION = 0.2
    DECELERATION = 0.35
    STOP_DISTANCE = 50

    GREEN_DURATION = 150
    YELLOW_DURATION = 50
    ALL_RED_DURATION = 30
    CONGESTION_THRESHOLD = 6
    HIGH_CONGESTION_THRESHOLD = 10

    INTERSECTION_HALF_WIDTH = 45
    YELLOW_EXTENSION = 15

    TURN_CURVE_POINTS = 15
    TURN_SPEED_FACTOR = 0.75

    INSTITUTION_ROUTE_WIDTH = 15
    # VERY CLOSE detection - only trigger when very near intersection
    EMERGENCY_DETECTION_DISTANCE = 80

    STATE_SIZE = 16
    ACTION_SIZE = 4
    MEMORY_SIZE = 20000
    BATCH_SIZE = 64
    GAMMA = 0.95
    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    LEARNING_RATE = 0.001

    MAX_EMERGENCY_VEHICLES = 3
    EMERGENCY_GENERATION_INTERVAL = 400
    EMERGENCY_OVERRIDE_DURATION = 25  # Longer override duration

    DB_NAME = 'traffic_emergency.db'
    SAVE_INTERVAL = 15


# ============================================
# ENUMS & DATA CLASSES
# ============================================

class Direction(Enum):
    NORTH = 'north'
    SOUTH = 'south'
    EAST = 'east'
    WEST = 'west'

    def get_arabic_name(self):
        names = {Direction.NORTH: 'شمال', Direction.SOUTH: 'جنوب', Direction.EAST: 'شرق', Direction.WEST: 'غرب'}
        return names.get(self, self.value)

    @classmethod
    def get_all(cls):
        return [cls.NORTH, cls.SOUTH, cls.EAST, cls.WEST]

    @classmethod
    def random(cls):
        return random.choice(cls.get_all())

    @classmethod
    def opposite(cls, direction):
        opposites = {cls.NORTH: cls.SOUTH, cls.SOUTH: cls.NORTH, cls.EAST: cls.WEST, cls.WEST: cls.EAST}
        return opposites.get(direction)


class LightState(Enum):
    GREEN = 'green'
    YELLOW = 'yellow'
    RED = 'red'
    ALL_RED = 'all_red'


class TurnDirection(Enum):
    STRAIGHT = 'straight'
    LEFT = 'left'
    RIGHT = 'right'


class VehicleStatus(Enum):
    MOVING = 'moving'
    STOPPED = 'stopped'
    WAITING = 'waiting'
    IN_INTERSECTION = 'in_intersection'
    CLEARING_INTERSECTION = 'clearing_intersection'
    PASSING = 'passing'
    TURNING = 'turning'
    ARRIVED = 'arrived'


@dataclass
class Position:
    x: float
    y: float

    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def as_tuple(self) -> Tuple[int, int]:
        return (int(self.x), int(self.y))

    def move_towards(self, target: 'Position', speed: float) -> 'Position':
        dx = target.x - self.x
        dy = target.y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        if dist < speed:
            return Position(target.x, target.y)
        return Position(self.x + (dx / dist) * speed, self.y + (dy / dist) * speed)


# ============================================
# EMERGENCY SERVICES SYSTEM
# ============================================

class GovernmentInstitution:
    def __init__(self, name: str, type: str, location: Tuple[int, int], contact: str,
                 route_to_main: List[Tuple[float, float]] = None, entry_direction: Direction = None):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.type = type
        self.location = Position(location[0], location[1])
        self.contact = contact
        self.vehicles: List['EmergencyVehicle'] = []
        self.status = 'active'
        self.emergencies_handled = 0
        self.route_to_main = route_to_main or []
        self.entry_direction = entry_direction
        self.icons = {'hospital': '🏥', 'fire': '🚒', 'police': '🚓', 'civil_defense': '🛡️'}
        self.icon = self.icons.get(type, '🏛️')
        self.colors = {'hospital': (0, 200, 100), 'fire': (255, 50, 0), 'police': (0, 50, 255), 'civil_defense': (255, 200, 0)}
        self.color = self.colors.get(type, (200, 200, 200))

    def get_distance_to(self, position: Position) -> float:
        return self.location.distance_to(position)


class EmergencyVehicle:
    def __init__(self, institution: GovernmentInstitution, destination: Position,
                 intersection_pos: Position, priority: int = 1):
        self.id = str(uuid.uuid4())[:8]
        self.institution = institution
        self.destination = destination
        self.intersection_pos = intersection_pos
        self.priority = priority
        self.type = institution.type
        self.position = Position(institution.location.x, institution.location.y)
        self.speed = Config.EMERGENCY_SPEED
        self.target_speed = Config.EMERGENCY_SPEED
        self.max_speed = Config.EMERGENCY_SPEED
        self.active = True
        self.arrived = False
        self.siren_on = True
        self.flashing_lights = True
        self.trail = deque(maxlen=20)
        self.status = 'moving_to_main_road'
        self.dispatch_time = datetime.now()

        # Route system
        self.route_points = []
        if institution.route_to_main:
            self.route_points = [Position(p[0], p[1]) for p in institution.route_to_main]
        self.current_route_index = 0

        # Movement state
        self.direction = None
        self.entry_direction = institution.entry_direction
        self.on_main_road = False
        self.passed_intersection = False
        self.approaching_intersection = False
        self.wait_time = 0
        self.stopped = False

        # Intersection behavior
        self.entered_intersection = False
        self.touching_intersection = False
        self.turn_decision = None
        self.turning = False
        self.new_direction = None
        self.turn_curve = []
        self.curve_index = 0
        self.lane_corrected = False

        # Visual
        self.icons = {'hospital': '🚑', 'fire': '🚒', 'police': '🚓', 'civil_defense': '🛡️'}
        self.icon = self.icons.get(self.type, '🚨')
        self.color = institution.color
        self.flash_timer = 0

        # Track which direction vehicle is coming from
        self.approach_direction = None

    def update_position(self, intersection_pos: Position, traffic_controller) -> bool:
        if self.arrived:
            return True

        self.flash_timer += 1

        # Stage 1: Moving from institution to main road (FAST)
        if not self.on_main_road and self.current_route_index < len(self.route_points):
            target = self.route_points[self.current_route_index]
            self.position = self.position.move_towards(target, self.speed)

            if self.current_route_index == len(self.route_points) - 1:
                if self.entry_direction:
                    target_x = self._get_lane_x(self.entry_direction)
                    target_y = self._get_lane_y(self.entry_direction)
                    if target_x is not None:
                        self.position.x += (target_x - self.position.x) * 0.15
                    if target_y is not None:
                        self.position.y += (target_y - self.position.y) * 0.15

            if self.position.distance_to(target) < 8:
                self.current_route_index += 1
                if self.current_route_index >= len(self.route_points):
                    self.on_main_road = True
                    self.direction = self.entry_direction
                    self.approach_direction = self.entry_direction
                    self.status = 'on_main_road'
                    self._snap_to_lane()

            self.trail.append(Position(self.position.x, self.position.y))
            return False

        # Stage 2: On main road - FAST with priority
        if self.on_main_road and not self.passed_intersection:
            return self._update_on_main_road(intersection_pos, traffic_controller)

        # Stage 3: After passing intersection
        if self.passed_intersection:
            self.position = self.position.move_towards(self.destination, self.speed)
            self.trail.append(Position(self.position.x, self.position.y))

            if self.direction:
                target_x = self._get_lane_x(self.direction)
                target_y = self._get_lane_y(self.direction)
                if target_x is not None:
                    diff = target_x - self.position.x
                    if abs(diff) > 1: self.position.x += diff * 0.2
                if target_y is not None:
                    diff = target_y - self.position.y
                    if abs(diff) > 1: self.position.y += diff * 0.2

            if self.position.distance_to(self.destination) < 8:
                self.arrived = True
                self.status = 'arrived'
                return True

        return False

    def _update_on_main_road(self, intersection_pos: Position, traffic_controller) -> bool:
        """Move FAST on main road - only trigger light when VERY close"""
        pos = self.position
        direction = self.direction
        ix, iy = intersection_pos.x, intersection_pos.y
        hw = Config.INTERSECTION_HALF_WIDTH

        in_square = (abs(pos.x - ix) <= hw and abs(pos.y - iy) <= hw)
        dist_to_center = pos.distance_to(intersection_pos)

        # ONLY trigger when VERY close to intersection
        if dist_to_center < Config.EMERGENCY_DETECTION_DISTANCE and not self.passed_intersection:
            if not self.approaching_intersection:
                self.approaching_intersection = True
                # Determine approach direction
                dx = pos.x - ix
                dy = pos.y - iy
                if abs(dx) > abs(dy):
                    self.approach_direction = Direction.WEST if dx < 0 else Direction.EAST
                else:
                    self.approach_direction = Direction.NORTH if dy < 0 else Direction.SOUTH

                # Only trigger override when very close
                if dist_to_center < Config.EMERGENCY_DETECTION_DISTANCE:
                    if not traffic_controller.priority_override_active:
                        traffic_controller.set_emergency_override(self.approach_direction, self)

        # Touching intersection zone
        touching_square = (
            (abs(pos.x - ix) <= hw + 10 and abs(pos.y - iy) <= hw + 10) and
            not (abs(pos.x - ix) <= hw - 10 and abs(pos.y - iy) <= hw - 10)
        )

        if touching_square and not in_square:
            self.touching_intersection = True
        elif in_square:
            self.touching_intersection = True

        # Choose random turn at intersection
        if dist_to_center < 80 and self.turn_decision is None:
            self.turn_decision = random.choice([TurnDirection.STRAIGHT, TurnDirection.LEFT, TurnDirection.RIGHT])
            if self.turn_decision != TurnDirection.STRAIGHT:
                self.new_direction = self._get_new_direction(direction, self.turn_decision)

        # Enter intersection
        if in_square and not self.entered_intersection:
            self.entered_intersection = True
            self.status = 'in_intersection'

            if self.turn_decision and self.turn_decision != TurnDirection.STRAIGHT:
                self.turning = True
                new_dir = self.new_direction
                end_x, end_y = pos.x, pos.y
                lw = Config.LANE_WIDTH
                if new_dir == Direction.NORTH:
                    end_x, end_y = ix - lw * 0.5, iy - hw - 25
                elif new_dir == Direction.SOUTH:
                    end_x, end_y = ix + lw * 0.5, iy + hw + 25
                elif new_dir == Direction.EAST:
                    end_x, end_y = ix + hw + 25, iy - lw * 0.5
                elif new_dir == Direction.WEST:
                    end_x, end_y = ix - hw - 25, iy + lw * 0.5

                self.turn_curve = self._generate_turn_curve(pos, Position(end_x, end_y), direction, new_dir)
                self.curve_index = 0

        # Exit intersection
        if self.entered_intersection and not in_square:
            if not self.passed_intersection:
                self.passed_intersection = True
                self.touching_intersection = False
                if self.turning and self.new_direction:
                    self.direction = self.new_direction
                    self.turning = False
                    self._snap_to_lane()
                    self.lane_corrected = True
                self.status = 'passed_intersection'

        # FAST movement
        move_speed = self.speed

        if self.entered_intersection and not self.passed_intersection:
            move_speed = self.speed * (Config.TURN_SPEED_FACTOR if self.turning else 0.85)

        # Handle turning
        if self.turning and self.turn_curve:
            curve = self.turn_curve
            idx = self.curve_index
            if idx < len(curve):
                target_point = curve[idx]
                dx = target_point.x - pos.x
                dy = target_point.y - pos.y
                dist_to_point = math.sqrt(dx**2 + dy**2)
                if dist_to_point < move_speed:
                    self.position.x = target_point.x
                    self.position.y = target_point.y
                    self.curve_index = idx + 1
                else:
                    self.position.x += (dx / dist_to_point) * move_speed
                    self.position.y += (dy / dist_to_point) * move_speed
            else:
                self.turning = False
                if self.new_direction:
                    self.direction = self.new_direction
                self._snap_to_lane()
                self.lane_corrected = True
        else:
            # Normal FAST movement
            if self.direction == Direction.NORTH:
                self.position.y -= move_speed
            elif self.direction == Direction.SOUTH:
                self.position.y += move_speed
            elif self.direction == Direction.EAST:
                self.position.x += move_speed
            elif self.direction == Direction.WEST:
                self.position.x -= move_speed

        self.trail.append(Position(self.position.x, self.position.y))
        return False

    def _get_lane_x(self, direction: Direction) -> float:
        ix = self.intersection_pos.x
        lw = Config.LANE_WIDTH
        if direction == Direction.NORTH: return ix - lw * 0.5
        elif direction == Direction.SOUTH: return ix + lw * 0.5
        return None

    def _get_lane_y(self, direction: Direction) -> float:
        iy = self.intersection_pos.y
        lw = Config.LANE_WIDTH
        if direction == Direction.EAST: return iy - lw * 0.5
        elif direction == Direction.WEST: return iy + lw * 0.5
        return None

    def _snap_to_lane(self):
        if self.direction:
            lane_x = self._get_lane_x(self.direction)
            lane_y = self._get_lane_y(self.direction)
            if lane_x is not None: self.position.x = lane_x
            if lane_y is not None: self.position.y = lane_y

    def _get_new_direction(self, current: Direction, turn: TurnDirection) -> Direction:
        if turn == TurnDirection.STRAIGHT: return current
        order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        idx = order.index(current)
        new_idx = (idx + 1) % 4 if turn == TurnDirection.RIGHT else (idx - 1) % 4
        return order[new_idx]

    def _generate_turn_curve(self, start: Position, end: Position, curr_dir: Direction, new_dir: Direction) -> List[Position]:
        ix, iy = self.intersection_pos.x, self.intersection_pos.y
        hw = Config.INTERSECTION_HALF_WIDTH

        turn_map = {
            (Direction.NORTH, Direction.EAST): (ix + hw * 0.8, iy - hw * 0.8),
            (Direction.NORTH, Direction.WEST): (ix - hw * 0.8, iy - hw * 0.8),
            (Direction.SOUTH, Direction.WEST): (ix - hw * 0.8, iy + hw * 0.8),
            (Direction.SOUTH, Direction.EAST): (ix + hw * 0.8, iy + hw * 0.8),
            (Direction.EAST, Direction.NORTH): (ix + hw * 0.8, iy - hw * 0.8),
            (Direction.EAST, Direction.SOUTH): (ix + hw * 0.8, iy + hw * 0.8),
            (Direction.WEST, Direction.NORTH): (ix - hw * 0.8, iy - hw * 0.8),
            (Direction.WEST, Direction.SOUTH): (ix - hw * 0.8, iy + hw * 0.8),
        }
        cx, cy = turn_map.get((curr_dir, new_dir), ((start.x + end.x) / 2, (start.y + end.y) / 2))

        points = []
        for i in range(Config.TURN_CURVE_POINTS + 1):
            t = i / Config.TURN_CURVE_POINTS
            x = (1-t)**2 * start.x + 2*(1-t)*t * cx + t**2 * end.x
            y = (1-t)**2 * start.y + 2*(1-t)*t * cy + t**2 * end.y
            points.append(Position(x, y))
        return points


class EmergencyDispatcher:
    def __init__(self, intersection_pos: Position):
        self.institutions: List[GovernmentInstitution] = []
        self.active_emergencies: List[Dict] = []
        self.emergency_history: List[Dict] = []
        self.alert_level = 0
        self.intersection_pos = intersection_pos
        self._init_institutions()

    def _init_institutions(self):
        ix, iy = self.intersection_pos.x, self.intersection_pos.y
        rw = Config.ROAD_WIDTH
        lw = Config.LANE_WIDTH

        institutions_data = [
            {
                'name': 'Central Peace Hospital', 'type': 'hospital',
                'location': (150, 150), 'contact': '011-1234567',
                'route': [(150, iy - rw//2), (ix - lw*0.5, iy - rw//2)],
                'entry_direction': Direction.NORTH
            },
            {
                'name': 'Emergency Medical Center', 'type': 'hospital',
                'location': (1050, 650), 'contact': '011-7654321',
                'route': [(1050, iy + rw//2), (ix + lw*0.5, iy + rw//2)],
                'entry_direction': Direction.SOUTH
            },
            {
                'name': 'Eastern Fire Station', 'type': 'fire',
                'location': (200, 700), 'contact': '012-3456789',
                'route': [(200, iy + rw//2), (ix - lw*0.5, iy + rw//2)],
                'entry_direction': Direction.SOUTH
            },
            {
                'name': 'Western Fire Station', 'type': 'fire',
                'location': (1000, 100), 'contact': '012-9876543',
                'route': [(1000, iy - rw//2), (ix + lw*0.5, iy - rw//2)],
                'entry_direction': Direction.NORTH
            },
            {
                'name': 'Traffic Police HQ', 'type': 'police',
                'location': (600, 50), 'contact': '013-4567890',
                'route': [(600, iy - rw//2), (ix, iy - rw//2)],
                'entry_direction': Direction.NORTH
            },
            {
                'name': 'Civil Defense Main Base', 'type': 'civil_defense',
                'location': (50, 400), 'contact': '014-5678901',
                'route': [(ix - rw//2, 400), (ix - rw//2, iy - lw*0.5)],
                'entry_direction': Direction.EAST
            }
        ]

        for data in institutions_data:
            route = data.pop('route')
            entry_dir = data.pop('entry_direction')
            inst = GovernmentInstitution(**data, route_to_main=route, entry_direction=entry_dir)
            self.institutions.append(inst)

    def dispatch_emergency(self, emergency_type: str, location: Position,
                          description: str = "", priority: int = 1) -> Optional[EmergencyVehicle]:
        suitable_inst = [i for i in self.institutions if i.type == emergency_type]
        if not suitable_inst:
            return None
        nearest = min(suitable_inst, key=lambda i: i.get_distance_to(location))
        vehicle = EmergencyVehicle(nearest, location, self.intersection_pos, priority)
        nearest.vehicles.append(vehicle)
        nearest.emergencies_handled += 1
        self.active_emergencies.append({
            'id': vehicle.id, 'type': emergency_type, 'location': (location.x, location.y),
            'description': description, 'priority': priority, 'institution': nearest.name, 'status': 'dispatched'
        })
        self.alert_level = min(5, self.alert_level + 1)
        return vehicle

    def complete_emergency(self, vehicle_id: str):
        for record in self.active_emergencies:
            if record['id'] == vehicle_id:
                record['status'] = 'completed'
                self.active_emergencies.remove(record)
                break


# ============================================
# TRAFFIC LIGHT CONTROLLER WITH CONGESTION PRIORITY
# ============================================

class TrafficLightController:
    def __init__(self, intersection_position: Position):
        self.intersection = intersection_position
        self.half_width = Config.INTERSECTION_HALF_WIDTH
        self.direction_sequence = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        self.current_direction_index = 0
        self.current_green_direction = self.direction_sequence[self.current_direction_index]
        self.phase_timer = 0
        self.current_state = 'green'
        self.green_duration = Config.GREEN_DURATION
        self.yellow_duration = Config.YELLOW_DURATION
        self.yellow_extension = Config.YELLOW_EXTENSION
        self.lights = {d: {'state': LightState.RED, 'emergency_override': False} for d in Direction.get_all()}
        self.lights[self.current_green_direction]['state'] = LightState.GREEN
        self.congestion_threshold = Config.CONGESTION_THRESHOLD
        self.high_congestion_threshold = Config.HIGH_CONGESTION_THRESHOLD
        self.priority_active = False
        self.priority_direction = None
        self.emergency_mode = False
        self.emergency_direction = None
        self.emergency_vehicle = None
        self.priority_override_active = False
        self.override_timer = 0
        self.override_duration = Config.EMERGENCY_OVERRIDE_DURATION * Config.FPS
        self.yellow_extension_active = False
        self.yellow_extension_timer = 0
        self.congestion_events = 0
        self.emergency_events = 0
        self.last_congestion_check = 0
        self.congestion_check_interval = 60  # Check congestion every 30 frames

    def get_vehicle_counts(self, vehicles: List[Dict]) -> Dict[Direction, int]:
        counts = {d: 0 for d in Direction.get_all()}
        for vehicle in vehicles:
            if not vehicle.get('passed_intersection', False):
                dist = vehicle['position'].distance_to(self.intersection)
                if dist < 200:
                    counts[vehicle['direction']] += 1
        return counts

    def is_position_in_square(self, pos: Position) -> bool:
        ix, iy = self.intersection.x, self.intersection.y
        hw = self.half_width
        return (abs(pos.x - ix) <= hw and abs(pos.y - iy) <= hw)

    def is_intersection_occupied(self, vehicles: List[Dict]) -> bool:
        for vehicle in vehicles:
            if self.is_position_in_square(vehicle['position']):
                if not vehicle.get('passed_intersection', False):
                    return True
        return False

    def get_most_congested_direction(self, vehicles: List[Dict]) -> Optional[Direction]:
        """Find the direction with most waiting vehicles"""
        counts = self.get_vehicle_counts(vehicles)
        max_count = 0
        max_direction = None

        for direction in Direction.get_all():
            if counts[direction] > max_count:
                max_count = counts[direction]
                max_direction = direction

        # Only switch if significantly more vehicles
        current_count = counts[self.current_green_direction]
        if max_direction and max_direction != self.current_green_direction:
            if max_count >= self.high_congestion_threshold:
                return max_direction
            if max_count >= self.congestion_threshold and max_count > current_count * 1.5:
                return max_direction

        return None

    def set_priority_direction(self, direction: Direction, reason: str):
        self.priority_active = True
        self.priority_direction = direction
        for d in self.lights:
            self.lights[d]['state'] = LightState.RED
        self.current_green_direction = direction
        self.current_direction_index = self.direction_sequence.index(direction)
        self.lights[direction]['state'] = LightState.GREEN
        self.current_state = 'green'
        self.phase_timer = 0
        self.green_duration = Config.GREEN_DURATION + 80  # Extra time for congested direction
        self.congestion_events += 1

    def set_emergency_override(self, direction: Direction, vehicle: EmergencyVehicle):
        if self.emergency_direction == direction and self.priority_override_active:
            return  # Already set for this direction

        for d in self.lights:
            self.lights[d]['state'] = LightState.RED
            self.lights[d]['emergency_override'] = False
        self.lights[direction]['state'] = LightState.GREEN
        self.lights[direction]['emergency_override'] = True
        self.emergency_mode = True
        self.emergency_direction = direction
        self.emergency_vehicle = vehicle
        self.priority_override_active = True
        self.override_timer = 0
        self.emergency_events += 1

    def update(self, vehicles: List[Dict], emergency_vehicles: List[EmergencyVehicle]):
        # Handle emergency override
        if self.priority_override_active:
            all_passed = all(ev.passed_intersection for ev in emergency_vehicles) if emergency_vehicles else True
            self.override_timer += 1
            if self.override_timer > self.override_duration or all_passed:
                self.clear_emergency_override()
            return

        if self.yellow_extension_active:
            self.yellow_extension_timer += 1
            if self.yellow_extension_timer >= self.yellow_extension:
                self.yellow_extension_active = False
                self._advance_to_next_direction()
            return

        # CONGESTION-BASED PRIORITY SYSTEM
        self.last_congestion_check += 1
        if self.last_congestion_check >= self.congestion_check_interval:
            self.last_congestion_check = 0
            most_congested = self.get_most_congested_direction(vehicles)
            if most_congested and not self.priority_active:
                counts = self.get_vehicle_counts(vehicles)
                self.set_priority_direction(most_congested,
                    f"Priority: {most_congested.get_arabic_name()} has {counts[most_congested]} vehicles")
                return

        self.phase_timer += 1
        if self.current_state == 'green':
            if self.phase_timer >= self.green_duration:
                self.current_state = 'yellow'
                self.phase_timer = 0
                for d in self.lights:
                    if self.lights[d]['state'] == LightState.GREEN:
                        self.lights[d]['state'] = LightState.YELLOW
        elif self.current_state == 'yellow':
            if self.phase_timer >= self.yellow_duration:
                if self.is_intersection_occupied(vehicles):
                    self.yellow_extension_active = True
                    self.yellow_extension_timer = 0
                else:
                    self._advance_to_next_direction()

    def _advance_to_next_direction(self):
        for d in self.lights:
            if not self.lights[d]['emergency_override']:
                self.lights[d]['state'] = LightState.RED

        # Check congestion before advancing to next in sequence
        # If another direction needs priority, skip to it

        self.current_direction_index = (self.current_direction_index + 1) % len(self.direction_sequence)
        next_direction = self.direction_sequence[self.current_direction_index]
        self.lights[next_direction]['state'] = LightState.GREEN
        self.current_green_direction = next_direction
        self.current_state = 'green'
        self.phase_timer = 0
        self.green_duration = Config.GREEN_DURATION
        self.priority_active = False

    def clear_emergency_override(self):
        for d in self.lights:
            self.lights[d]['emergency_override'] = False
        self.emergency_mode = False
        self.priority_override_active = False
        self.override_timer = 0
        self.emergency_direction = None
        self.emergency_vehicle = None
        for d in self.lights:
            self.lights[d]['state'] = LightState.RED
        self.lights[self.current_green_direction]['state'] = LightState.GREEN
        self.current_state = 'green'
        self.phase_timer = 0

    def get_light_state(self, direction: Direction) -> LightState:
        return self.lights[direction]['state']

    def is_green(self, direction: Direction) -> bool:
        return self.lights[direction]['state'] == LightState.GREEN

    def is_red(self, direction: Direction) -> bool:
        return self.lights[direction]['state'] in [LightState.RED, LightState.ALL_RED]

    def is_yellow(self, direction: Direction) -> bool:
        return self.lights[direction]['state'] == LightState.YELLOW

    def get_current_phase_name(self) -> str:
        if self.emergency_mode:
            return f"🚨 EMERGENCY {self.emergency_direction.get_arabic_name() if self.emergency_direction else ''}"
        if self.priority_active:
            return f"⭐ PRIORITY {self.current_green_direction.get_arabic_name()}"
        state_map = {'green': f'🟢 {self.current_green_direction.get_arabic_name()}',
                    'yellow': f'🟡 {self.current_green_direction.get_arabic_name()}'}
        return f"{state_map.get(self.current_state, '?')} ({self.phase_timer})"

    def get_green_directions(self) -> List[str]:
        return [d.get_arabic_name() for d, l in self.lights.items() if l['state'] == LightState.GREEN]


# ============================================
# AI CONTROLLER (DQN)
# ============================================

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)


class TrafficAIAgent:
    def __init__(self, state_size: int = Config.STATE_SIZE, action_size: int = Config.ACTION_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        self.epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.gamma = Config.GAMMA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.training_steps = 0
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, emergency_mode=False):
        if emergency_mode or np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.model.eval()
            q_values = self.model(state_tensor)
            self.model.train()
        return int(np.argmax(q_values.cpu().numpy()[0]))

    def replay(self, batch_size=Config.BATCH_SIZE):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in minibatch])).to(self.device)
        dones = torch.BoolTensor(np.array([e[4] for e in minibatch])).to(self.device)
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * ~dones)
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.training_steps += 1
        if self.training_steps % 100 == 0:
            self.update_target_model()


# ============================================
# MAIN TRAFFIC SIMULATOR
# ============================================

class TrafficSimulator:
    def __init__(self):
        pygame.init()
        self.width = Config.SCREEN_WIDTH
        self.height = Config.SCREEN_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("🚦 Traffic System v6.2 - Fast Emergency + Smart Priority")
        self.clock = pygame.time.Clock()
        self.running = True
        self.fps = Config.FPS

        self.intersection = Position(self.width // 2, self.height // 2)
        self.road_width = Config.ROAD_WIDTH
        self.lane_width = Config.LANE_WIDTH
        self.stop_distance = Config.STOP_DISTANCE
        self.half_width = Config.INTERSECTION_HALF_WIDTH

        self.traffic_controller = TrafficLightController(self.intersection)
        self.dispatcher = EmergencyDispatcher(self.intersection)
        self.ai_agent = TrafficAIAgent()
        self.ai_active = False

        self.vehicles: List[Dict] = []
        self.emergency_vehicles: List[EmergencyVehicle] = []
        self.spawn_timer = 0
        self.emergency_timer = 0
        self.emergency_interval = Config.EMERGENCY_GENERATION_INTERVAL

        self.metrics = {'total_vehicles': 0, 'avg_wait_time': 0, 'throughput': 0, 'collisions': 0,
                       'emergency_completed': 0, 'total_emergencies': 0}

        self.COLORS = {
            'road': (50, 50, 60), 'lane_marking': (255, 255, 255), 'sidewalk': (80, 80, 90),
            'grass': (34, 139, 34), 'stop_line': (255, 255, 255), 'intersection_zone': (100, 100, 120),
            'intersection_border': (200, 200, 100), 'institution_route': (80, 80, 100),
        }

        self.car_colors = [
            (220, 20, 60), (0, 100, 200), (50, 205, 50), (255, 215, 0), (138, 43, 226),
            (255, 140, 0), (0, 139, 139), (199, 21, 133), (70, 130, 180), (160, 82, 45)
        ]

        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 16)

    # ============================================
    # LANE SYSTEM
    # ============================================

    def get_correct_lane_x(self, direction: Direction) -> float:
        ix = self.intersection.x
        lw = self.lane_width
        if direction == Direction.NORTH: return ix - lw * 0.5
        elif direction == Direction.SOUTH: return ix + lw * 0.5
        return None

    def get_correct_lane_y(self, direction: Direction) -> float:
        iy = self.intersection.y
        lw = self.lane_width
        if direction == Direction.EAST: return iy - lw * 0.5
        elif direction == Direction.WEST: return iy + lw * 0.5
        return None

    def snap_to_correct_lane(self, vehicle: Dict):
        direction = vehicle['direction']
        lane_x = self.get_correct_lane_x(direction)
        lane_y = self.get_correct_lane_y(direction)
        if lane_x is not None: vehicle['position'].x = lane_x
        if lane_y is not None: vehicle['position'].y = lane_y

    def get_lane_position(self, direction: Direction, lane_type: str) -> Tuple[float, float]:
        ix, iy = self.intersection.x, self.intersection.y
        lw = self.lane_width
        if direction == Direction.NORTH:
            return ix - lw * 0.5, (self.height + 20 if lane_type == 'going' else -20)
        elif direction == Direction.SOUTH:
            return ix + lw * 0.5, (-20 if lane_type == 'going' else self.height + 20)
        elif direction == Direction.EAST:
            return (-20 if lane_type == 'going' else self.width + 20), iy - lw * 0.5
        else:
            return (self.width + 20 if lane_type == 'going' else -20), iy + lw * 0.5

    def generate_turn_curve(self, start: Position, end: Position, current_dir: Direction, new_dir: Direction) -> List[Position]:
        ix, iy = self.intersection.x, self.intersection.y
        hw = self.half_width

        turn_map = {
            (Direction.NORTH, Direction.EAST): (ix + hw * 0.8, iy - hw * 0.8),
            (Direction.NORTH, Direction.WEST): (ix - hw * 0.8, iy - hw * 0.8),
            (Direction.SOUTH, Direction.WEST): (ix - hw * 0.8, iy + hw * 0.8),
            (Direction.SOUTH, Direction.EAST): (ix + hw * 0.8, iy + hw * 0.8),
            (Direction.EAST, Direction.NORTH): (ix + hw * 0.8, iy - hw * 0.8),
            (Direction.EAST, Direction.SOUTH): (ix + hw * 0.8, iy + hw * 0.8),
            (Direction.WEST, Direction.NORTH): (ix - hw * 0.8, iy - hw * 0.8),
            (Direction.WEST, Direction.SOUTH): (ix - hw * 0.8, iy + hw * 0.8),
        }
        cx, cy = turn_map.get((current_dir, new_dir), ((start.x + end.x) / 2, (start.y + end.y) / 2))

        points = []
        for i in range(Config.TURN_CURVE_POINTS + 1):
            t = i / Config.TURN_CURVE_POINTS
            x = (1-t)**2 * start.x + 2*(1-t)*t * cx + t**2 * end.x
            y = (1-t)**2 * start.y + 2*(1-t)*t * cy + t**2 * end.y
            points.append(Position(x, y))
        return points

    def is_position_in_square(self, pos: Position) -> bool:
        ix, iy = self.intersection.x, self.intersection.y
        hw = self.half_width
        return (abs(pos.x - ix) <= hw and abs(pos.y - iy) <= hw)

    def get_random_turn(self, direction: Direction) -> TurnDirection:
        rand = random.random()
        if rand < 0.5: return TurnDirection.STRAIGHT
        elif rand < 0.75: return TurnDirection.LEFT
        else: return TurnDirection.RIGHT

    def get_new_direction_after_turn(self, current_direction: Direction, turn: TurnDirection) -> Direction:
        if turn == TurnDirection.STRAIGHT: return current_direction
        direction_order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        current_index = direction_order.index(current_direction)
        new_index = (current_index + 1) % 4 if turn == TurnDirection.RIGHT else (current_index - 1) % 4
        return direction_order[new_index]

    def spawn_vehicle(self):
        if len(self.vehicles) >= Config.MAX_VEHICLES:
            return
        direction = Direction.random()
        lane_type = random.choice(['going', 'returning'])
        x, y = self.get_lane_position(direction, lane_type)
        base_speed = random.uniform(Config.NORMAL_SPEED_MIN, Config.NORMAL_SPEED_MAX)
        vehicle = {
            'id': str(uuid.uuid4())[:8],
            'position': Position(x, y),
            'direction': direction,
            'speed': abs(base_speed),
            'target_speed': abs(base_speed),
            'max_speed': abs(base_speed),
            'status': VehicleStatus.MOVING,
            'wait_time': 0,
            'color': random.choice(self.car_colors),
            'stopped': False,
            'passed_intersection': False,
            'entered_intersection': False,
            'touching_intersection': False,
            'turn_decision': None,
            'turning': False,
            'new_direction': None,
            'turn_curve': [],
            'curve_index': 0,
            'lane_corrected': False,
        }
        self.vehicles.append(vehicle)
        self.metrics['total_vehicles'] += 1

    # ============================================
    # VEHICLE UPDATES
    # ============================================

    def update_vehicles(self):
        for vehicle in self.vehicles[:]:
            pos = vehicle['position']
            direction = vehicle['direction']
            light_state = self.traffic_controller.get_light_state(direction)
            in_square = self.is_position_in_square(pos)
            ix, iy = self.intersection.x, self.intersection.y
            hw = self.half_width

            touching_square = (
                (abs(pos.x - ix) <= hw + 10 and abs(pos.y - iy) <= hw + 10) and
                not (abs(pos.x - ix) <= hw - 10 and abs(pos.y - iy) <= hw - 10)
            )

            if touching_square and not in_square:
                vehicle['touching_intersection'] = True
            elif in_square:
                vehicle['touching_intersection'] = True

            dist_to_center = pos.distance_to(self.intersection)
            if dist_to_center < 100 and vehicle.get('turn_decision') is None:
                vehicle['turn_decision'] = self.get_random_turn(direction)
                if vehicle['turn_decision'] != TurnDirection.STRAIGHT:
                    vehicle['new_direction'] = self.get_new_direction_after_turn(direction, vehicle['turn_decision'])

            if in_square and not vehicle.get('entered_intersection'):
                vehicle['entered_intersection'] = True
                vehicle['status'] = VehicleStatus.IN_INTERSECTION

                if vehicle.get('turn_decision') and vehicle['turn_decision'] != TurnDirection.STRAIGHT:
                    vehicle['turning'] = True
                    vehicle['status'] = VehicleStatus.TURNING
                    new_dir = vehicle['new_direction']
                    end_x, end_y = pos.x, pos.y
                    if new_dir == Direction.NORTH:
                        end_x, end_y = ix - self.lane_width * 0.5, iy - hw - 25
                    elif new_dir == Direction.SOUTH:
                        end_x, end_y = ix + self.lane_width * 0.5, iy + hw + 25
                    elif new_dir == Direction.EAST:
                        end_x, end_y = ix + hw + 25, iy - self.lane_width * 0.5
                    elif new_dir == Direction.WEST:
                        end_x, end_y = ix - hw - 25, iy + self.lane_width * 0.5

                    vehicle['turn_curve'] = self.generate_turn_curve(pos, Position(end_x, end_y), direction, new_dir)
                    vehicle['curve_index'] = 0

            if vehicle.get('entered_intersection') and not in_square:
                if not vehicle.get('passed_intersection'):
                    vehicle['passed_intersection'] = True
                    vehicle['touching_intersection'] = False
                    if vehicle.get('turning') and vehicle.get('new_direction'):
                        vehicle['direction'] = vehicle['new_direction']
                        vehicle['turning'] = False
                        self.snap_to_correct_lane(vehicle)
                        vehicle['lane_corrected'] = True
                    vehicle['status'] = VehicleStatus.PASSING

            should_stop = False

            if vehicle.get('entered_intersection') and not vehicle.get('passed_intersection'):
                should_stop = False
                vehicle['target_speed'] = vehicle['max_speed'] * (Config.TURN_SPEED_FACTOR if vehicle.get('turning') else 0.7)

            elif vehicle.get('touching_intersection') and not vehicle.get('passed_intersection'):
                if self.traffic_controller.is_yellow(direction) or self.traffic_controller.is_red(direction):
                    should_stop = False
                    vehicle['target_speed'] = vehicle['max_speed']
                    vehicle['entered_intersection'] = True
                    vehicle['status'] = VehicleStatus.CLEARING_INTERSECTION

            elif not vehicle.get('passed_intersection', False):
                if dist_to_center < 100:
                    if self.traffic_controller.is_red(direction):
                        should_stop = dist_to_center > self.half_width + 15
                        if not should_stop:
                            vehicle['entered_intersection'] = True
                    elif self.traffic_controller.is_yellow(direction):
                        if dist_to_center < self.half_width + 20:
                            should_stop = False
                            vehicle['entered_intersection'] = True
                            vehicle['status'] = VehicleStatus.CLEARING_INTERSECTION
                        elif dist_to_center < self.stop_distance + 10:
                            should_stop = random.random() >= 0.4
                            if not should_stop:
                                vehicle['entered_intersection'] = True
                        else:
                            should_stop = True

                    if self.traffic_controller.emergency_mode:
                        if direction != self.traffic_controller.emergency_direction:
                            if dist_to_center < 100 and not vehicle.get('entered_intersection'):
                                should_stop = True

            if not vehicle.get('entered_intersection'):
                for other in self.vehicles:
                    if other['id'] != vehicle['id'] and other['direction'] == direction:
                        dist = pos.distance_to(other['position'])
                        if dist < 25:
                            is_ahead = ((direction == Direction.NORTH and other['position'].y < pos.y) or
                                       (direction == Direction.SOUTH and other['position'].y > pos.y) or
                                       (direction == Direction.EAST and other['position'].x > pos.x) or
                                       (direction == Direction.WEST and other['position'].x < pos.x))
                            if is_ahead:
                                should_stop = True

            if should_stop:
                vehicle['target_speed'] = 0
                vehicle['stopped'] = True
                vehicle['wait_time'] += 1 / Config.FPS
            elif not vehicle.get('entered_intersection'):
                vehicle['target_speed'] = vehicle['max_speed']
                vehicle['stopped'] = False

            if vehicle['speed'] < vehicle['target_speed']:
                vehicle['speed'] = min(vehicle['target_speed'], vehicle['speed'] + Config.ACCELERATION)
            elif vehicle['speed'] > vehicle['target_speed']:
                vehicle['speed'] = max(vehicle['target_speed'], vehicle['speed'] - Config.DECELERATION)

            move_speed = vehicle['speed']

            if vehicle.get('turning') and vehicle.get('turn_curve'):
                curve = vehicle['turn_curve']
                idx = vehicle.get('curve_index', 0)
                if idx < len(curve):
                    target_point = curve[idx]
                    dx = target_point.x - pos.x
                    dy = target_point.y - pos.y
                    dist_to_point = math.sqrt(dx**2 + dy**2)
                    if dist_to_point < move_speed:
                        vehicle['position'].x = target_point.x
                        vehicle['position'].y = target_point.y
                        vehicle['curve_index'] = idx + 1
                    else:
                        vehicle['position'].x += (dx / dist_to_point) * move_speed
                        vehicle['position'].y += (dy / dist_to_point) * move_speed
                else:
                    vehicle['turning'] = False
                    if vehicle.get('new_direction'):
                        vehicle['direction'] = vehicle['new_direction']
                    self.snap_to_correct_lane(vehicle)
                    vehicle['lane_corrected'] = True
            else:
                if direction == Direction.NORTH: vehicle['position'].y -= move_speed
                elif direction == Direction.SOUTH: vehicle['position'].y += move_speed
                elif direction == Direction.EAST: vehicle['position'].x += move_speed
                elif direction == Direction.WEST: vehicle['position'].x -= move_speed

                if not in_square and vehicle.get('lane_corrected'):
                    target_x = self.get_correct_lane_x(direction)
                    target_y = self.get_correct_lane_y(direction)
                    if target_x is not None:
                        diff = target_x - vehicle['position'].x
                        if abs(diff) > 1: vehicle['position'].x += diff * 0.2
                    if target_y is not None:
                        diff = target_y - vehicle['position'].y
                        if abs(diff) > 1: vehicle['position'].y += diff * 0.2

            margin = 50
            pos = vehicle['position']
            if (pos.y < -margin or pos.y > self.height + margin or
                pos.x < -margin or pos.x > self.width + margin):
                self.vehicles.remove(vehicle)
                self.metrics['throughput'] += 1
                if vehicle['wait_time'] > 0:
                    self.metrics['avg_wait_time'] = (self.metrics['avg_wait_time'] * 0.95 + vehicle['wait_time'] * 0.05)

    # ============================================
    # EMERGENCY MANAGEMENT
    # ============================================

    def generate_emergency_event(self):
        if len(self.emergency_vehicles) >= Config.MAX_EMERGENCY_VEHICLES:
            return
        emergency_types = ['hospital', 'fire', 'police', 'civil_defense']
        emergency_type = random.choice(emergency_types)
        location = Position(
            self.intersection.x + random.randint(-300, 300),
            self.intersection.y + random.randint(-300, 300)
        )
        vehicle = self.dispatcher.dispatch_emergency(emergency_type, location)
        if vehicle:
            self.emergency_vehicles.append(vehicle)
            self.metrics['total_emergencies'] += 1

    def update_emergency_vehicles(self):
        for vehicle in self.emergency_vehicles[:]:
            arrived = vehicle.update_position(self.intersection, self.traffic_controller)
            if vehicle.arrived:
                self.metrics['emergency_completed'] += 1
                self.dispatcher.complete_emergency(vehicle.id)
                self.emergency_vehicles.remove(vehicle)

    # ============================================
    # RENDERING
    # ============================================

    def draw(self):
        self.screen.fill(self.COLORS['grass'])
        self._draw_institution_routes()
        self._draw_roads()
        self._draw_square_intersection_zone()
        self._draw_stop_lines()
        self._draw_traffic_lights()
        self._draw_institutions()
        self._draw_vehicles()
        self._draw_emergency_vehicles()
        self._draw_hud()
        self._draw_congestion_info()
        self._draw_emergency_alert()
        pygame.display.flip()

    def _draw_institution_routes(self):
        route_width = Config.INSTITUTION_ROUTE_WIDTH
        route_color = self.COLORS['institution_route']

        for inst in self.dispatcher.institutions:
            if inst.route_to_main:
                points = [inst.location.as_tuple()] + [(int(p[0]), int(p[1])) for p in inst.route_to_main]
                if len(points) >= 2:
                    for i in range(len(points) - 1):
                        pygame.draw.line(self.screen, route_color, points[i], points[i+1], route_width)
                        pygame.draw.line(self.screen, (120, 120, 140), points[i], points[i+1], route_width + 2)
                        pygame.draw.line(self.screen, route_color, points[i], points[i+1], route_width)

    def _draw_roads(self):
        ix, iy = self.intersection.x, self.intersection.y
        rw = self.road_width
        lw = self.lane_width

        pygame.draw.rect(self.screen, self.COLORS['road'], (0, iy - rw // 2, self.width, rw))
        pygame.draw.rect(self.screen, self.COLORS['road'], (ix - rw // 2, 0, rw, self.height))

        sidewalk_color = self.COLORS['sidewalk']
        pygame.draw.rect(self.screen, sidewalk_color, (0, iy - rw // 2 - 5, self.width, 5))
        pygame.draw.rect(self.screen, sidewalk_color, (0, iy + rw // 2, self.width, 5))
        pygame.draw.rect(self.screen, sidewalk_color, (ix - rw // 2 - 5, 0, 5, self.height))
        pygame.draw.rect(self.screen, sidewalk_color, (ix + rw // 2, 0, 5, self.height))

        pygame.draw.line(self.screen, self.COLORS['lane_marking'], (0, iy), (self.width, iy), 3)
        pygame.draw.line(self.screen, self.COLORS['lane_marking'], (ix, 0), (ix, self.height), 3)

        lane_color = self.COLORS['lane_marking']
        for offset in [-lw, lw]:
            y = iy + offset
            if abs(y - iy) > 15:
                for x in range(0, self.width, 40):
                    if abs(x - ix) > 30:
                        pygame.draw.line(self.screen, lane_color, (x, y), (x + 20, y), 2)
        for offset in [-lw, lw]:
            x = ix + offset
            if abs(x - ix) > 15:
                for y in range(0, self.height, 40):
                    if abs(y - iy) > 30:
                        pygame.draw.line(self.screen, lane_color, (x, y), (x, y + 20), 2)

    def _draw_square_intersection_zone(self):
        ix, iy = self.intersection.x, self.intersection.y
        hw = self.half_width
        zone_surface = pygame.Surface((hw * 2, hw * 2))
        zone_surface.set_alpha(40)
        zone_surface.fill(self.COLORS['intersection_zone'])
        self.screen.blit(zone_surface, (ix - hw, iy - hw))
        pygame.draw.rect(self.screen, self.COLORS['intersection_border'], (ix - hw, iy - hw, hw * 2, hw * 2), 3)
        pygame.draw.rect(self.screen, (255, 255, 0, 80), (ix - hw - 10, iy - hw - 10, (hw + 10) * 2, (hw + 10) * 2), 1)

    def _draw_stop_lines(self):
        ix, iy = self.intersection.x, self.intersection.y
        stop_dist = self.stop_distance
        lw = self.lane_width
        for y_off in [iy - stop_dist, iy + stop_dist]:
            pygame.draw.line(self.screen, self.COLORS['stop_line'], (ix - lw, y_off), (ix + lw, y_off), 4)
        for x_off in [ix - stop_dist, ix + stop_dist]:
            pygame.draw.line(self.screen, self.COLORS['stop_line'], (x_off, iy - lw), (x_off, iy + lw), 4)

    def _draw_traffic_lights(self):
        ix, iy = self.intersection.x, self.intersection.y
        rw = self.road_width
        light_positions = {
            Direction.NORTH: (ix, iy - rw // 2 - 20),
            Direction.SOUTH: (ix, iy + rw // 2 + 20),
            Direction.EAST: (ix + rw // 2 + 20, iy),
            Direction.WEST: (ix - rw // 2 - 20, iy)
        }
        for direction, pos in light_positions.items():
            light = self.traffic_controller.lights[direction]
            state = light['state']
            pygame.draw.rect(self.screen, (60, 60, 60), (pos[0] - 8, pos[1] - 30, 16, 60))

            r_color = (255, 0, 0) if state in [LightState.RED, LightState.ALL_RED] else (80, 0, 0)
            y_color = (255, 255, 0) if state == LightState.YELLOW else (80, 80, 0)
            g_color = (0, 255, 0) if state == LightState.GREEN else (0, 80, 0)

            pygame.draw.circle(self.screen, r_color, (pos[0], pos[1] - 15), 6)
            pygame.draw.circle(self.screen, y_color, (pos[0], pos[1]), 6)
            pygame.draw.circle(self.screen, g_color, (pos[0], pos[1] + 15), 6)

            if light['emergency_override']:
                if int(pygame.time.get_ticks() / 200) % 2 == 0:
                    pygame.draw.circle(self.screen, (255, 255, 255), pos, 20, 2)

            dir_label = self.font_small.render(direction.value[:1].upper(), True, (200, 200, 200))
            self.screen.blit(dir_label, (pos[0] - 4, pos[1] - 45))

    def _draw_institutions(self):
        for inst in self.dispatcher.institutions:
            x, y = int(inst.location.x), int(inst.location.y)
            pygame.draw.rect(self.screen, inst.color, (x - 20, y - 15, 40, 30))
            pygame.draw.rect(self.screen, (255, 255, 255), (x - 18, y - 13, 36, 26), 1)
            label = self.font_medium.render(inst.icon, True, (255, 255, 255))
            self.screen.blit(label, (x - 12, y - 10))
            name_label = self.font_small.render(inst.name[:15], True, (255, 255, 255))
            self.screen.blit(name_label, (x - 30, y + 20))

    def _draw_vehicles(self):
        for vehicle in self.vehicles:
            x, y = int(vehicle['position'].x), int(vehicle['position'].y)
            color = vehicle['color']
            direction = vehicle['direction']

            display_direction = vehicle.get('new_direction') if vehicle.get('turning') else direction

            if vehicle.get('entered_intersection') and not vehicle.get('passed_intersection'):
                pygame.draw.rect(self.screen, (0, 255, 255), (x - 12, y - 12, 24, 24), 2)

            if vehicle.get('touching_intersection') and not vehicle.get('passed_intersection'):
                pygame.draw.circle(self.screen, (255, 255, 0), (x, y), 18, 1)

            if vehicle.get('turn_decision') and not vehicle.get('passed_intersection'):
                turn_data = {TurnDirection.LEFT: ("L", (255, 100, 100)),
                           TurnDirection.RIGHT: ("R", (100, 255, 100)),
                           TurnDirection.STRAIGHT: ("S", (100, 100, 255))}
                turn_text, turn_color = turn_data.get(vehicle['turn_decision'], ("?", (255, 255, 255)))
                self.screen.blit(self.font_small.render(turn_text, True, turn_color), (x - 6, y - 25))

            if display_direction in [Direction.NORTH, Direction.SOUTH]:
                pygame.draw.rect(self.screen, color, (x - 8, y - 14, 16, 28))
                pygame.draw.rect(self.screen, (200, 220, 255), (x - 6, y - 10, 12, 16))
                if display_direction == Direction.NORTH:
                    pygame.draw.circle(self.screen, (255, 255, 200), (x - 5, y - 12), 3)
                    pygame.draw.circle(self.screen, (255, 255, 200), (x + 5, y - 12), 3)
                    pygame.draw.circle(self.screen, (255, 50, 50), (x - 5, y + 12), 2)
                    pygame.draw.circle(self.screen, (255, 50, 50), (x + 5, y + 12), 2)
                else:
                    pygame.draw.circle(self.screen, (255, 255, 200), (x - 5, y + 12), 3)
                    pygame.draw.circle(self.screen, (255, 255, 200), (x + 5, y + 12), 3)
                    pygame.draw.circle(self.screen, (255, 50, 50), (x - 5, y - 12), 2)
                    pygame.draw.circle(self.screen, (255, 50, 50), (x + 5, y - 12), 2)
            else:
                pygame.draw.rect(self.screen, color, (x - 14, y - 8, 28, 16))
                pygame.draw.rect(self.screen, (200, 220, 255), (x - 10, y - 6, 16, 12))
                if display_direction == Direction.EAST:
                    pygame.draw.circle(self.screen, (255, 255, 200), (x + 12, y - 5), 3)
                    pygame.draw.circle(self.screen, (255, 255, 200), (x + 12, y + 5), 3)
                    pygame.draw.circle(self.screen, (255, 50, 50), (x - 12, y - 5), 2)
                    pygame.draw.circle(self.screen, (255, 50, 50), (x - 12, y + 5), 2)
                else:
                    pygame.draw.circle(self.screen, (255, 255, 200), (x - 12, y - 5), 3)
                    pygame.draw.circle(self.screen, (255, 255, 200), (x - 12, y + 5), 3)
                    pygame.draw.circle(self.screen, (255, 50, 50), (x + 12, y - 5), 2)
                    pygame.draw.circle(self.screen, (255, 50, 50), (x + 12, y + 5), 2)

            if vehicle['wait_time'] > 3:
                pygame.draw.circle(self.screen, (255, min(255, int(vehicle['wait_time'] * 30)), 0), (x, y - 22), 4)

    def _draw_emergency_vehicles(self):
        for vehicle in self.emergency_vehicles:
            x, y = int(vehicle.position.x), int(vehicle.position.y)
            color = vehicle.color
            direction = vehicle.direction

            display_direction = vehicle.new_direction if vehicle.turning and vehicle.new_direction else direction

            if vehicle.flashing_lights and int(pygame.time.get_ticks() / 150) % 2 == 0:
                pygame.draw.circle(self.screen, (255, 0, 0), (x, y), 22, 2)
                pygame.draw.circle(self.screen, (0, 0, 255), (x, y), 18, 2)

            if display_direction in [Direction.NORTH, Direction.SOUTH]:
                pygame.draw.rect(self.screen, (255, 255, 255), (x - 8, y - 14, 16, 28))
                pygame.draw.rect(self.screen, color, (x - 8, y - 14, 16, 5))
                pygame.draw.rect(self.screen, color, (x - 8, y + 9, 16, 5))
                pygame.draw.rect(self.screen, (50, 50, 80), (x - 6, y - 6, 12, 12))

                if display_direction == Direction.NORTH:
                    pygame.draw.circle(self.screen, (255, 255, 200), (x - 5, y - 10), 3)
                    pygame.draw.circle(self.screen, (255, 255, 200), (x + 5, y - 10), 3)
                else:
                    pygame.draw.circle(self.screen, (255, 255, 200), (x - 5, y + 10), 3)
                    pygame.draw.circle(self.screen, (255, 255, 200), (x + 5, y + 10), 3)
            else:
                pygame.draw.rect(self.screen, (255, 255, 255), (x - 14, y - 8, 28, 16))
                pygame.draw.rect(self.screen, color, (x - 14, y - 8, 5, 16))
                pygame.draw.rect(self.screen, color, (x + 9, y - 8, 5, 16))
                pygame.draw.rect(self.screen, (50, 50, 80), (x - 6, y - 6, 12, 12))

                if display_direction == Direction.EAST:
                    pygame.draw.circle(self.screen, (255, 255, 200), (x + 10, y - 5), 3)
                    pygame.draw.circle(self.screen, (255, 255, 200), (x + 10, y + 5), 3)
                else:
                    pygame.draw.circle(self.screen, (255, 255, 200), (x - 10, y - 5), 3)
                    pygame.draw.circle(self.screen, (255, 255, 200), (x - 10, y + 5), 3)

            if int(pygame.time.get_ticks() / 100) % 2 == 0:
                siren_color = (255, 0, 0)
            else:
                siren_color = (0, 0, 255)
            pygame.draw.circle(self.screen, siren_color, (x, y - 16), 3)

            label = self.font_small.render(vehicle.icon, True, (255, 255, 255))
            self.screen.blit(label, (x - 8, y - 28))

    def _draw_hud(self):
        hud_surface = pygame.Surface((400, 460))
        hud_surface.set_alpha(180)
        hud_surface.fill((0, 0, 0))
        self.screen.blit(hud_surface, (10, 10))

        counts = self.traffic_controller.get_vehicle_counts(self.vehicles)
        green_dirs = self.traffic_controller.get_green_directions()

        info = [
            f"🚗 v6.2 Fast EV + Smart Priority",
            f"Vehicles: {len(self.vehicles)}/{Config.MAX_VEHICLES} | Thru: {self.metrics['throughput']}",
            f"Phase: {self.traffic_controller.get_current_phase_name()}",
            f"Green: {', '.join(green_dirs) if green_dirs else 'None'}",
            f"N:{counts[Direction.NORTH]} S:{counts[Direction.SOUTH]} E:{counts[Direction.EAST]} W:{counts[Direction.WEST]}",
            f"Emergency: {len(self.emergency_vehicles)} | Done: {self.metrics['emergency_completed']}",
            f"EV Speed: {Config.EMERGENCY_SPEED} | Detect: {Config.EMERGENCY_DETECTION_DISTANCE}px",
            f"Priority: Most congested direction first",
            f"FPS: {int(self.clock.get_fps())}"
        ]

        y_pos = 20
        for text in info:
            self.screen.blit(self.font_medium.render(text, True, (255, 255, 255)), (20, y_pos))
            y_pos += 28

        self.screen.blit(self.font_small.render("SPACE:AI | R:Reset | E:Emergency | ESC:Exit", True, (150, 150, 150)),
                        (20, self.height - 30))

    def _draw_congestion_info(self):
        counts = self.traffic_controller.get_vehicle_counts(self.vehicles)
        directions_info = [
            (Direction.NORTH, self.intersection.x, self.intersection.y - self.road_width - 40),
            (Direction.SOUTH, self.intersection.x, self.intersection.y + self.road_width + 40),
            (Direction.EAST, self.intersection.x + self.road_width + 40, self.intersection.y),
            (Direction.WEST, self.intersection.x - self.road_width - 40, self.intersection.y)
        ]
        for direction, x, y in directions_info:
            count = counts[direction]
            color = (255, 0, 0) if count >= Config.HIGH_CONGESTION_THRESHOLD else \
                   (255, 165, 0) if count >= Config.CONGESTION_THRESHOLD else (0, 255, 0)
            self.screen.blit(self.font_small.render(str(count), True, color), (x - 8, y - 8))

    def _draw_emergency_alert(self):
        if self.traffic_controller.emergency_mode:
            if int(pygame.time.get_ticks() / 500) % 2 == 0:
                alert = self.font_large.render("🚨 EMERGENCY - OPENING INTERSECTION 🚨", True, (255, 0, 0))
                rect = alert.get_rect(center=(self.width // 2, 30))
                pygame.draw.rect(self.screen, (0, 0, 0, 200), (rect.x - 15, rect.y - 8, rect.width + 30, rect.height + 16))
                self.screen.blit(alert, rect)
        elif self.traffic_controller.priority_active:
            alert = self.font_medium.render(f"⭐ PRIORITY: {self.traffic_controller.priority_direction.get_arabic_name() if self.traffic_controller.priority_direction else ''} - HIGH CONGESTION", True, (255, 165, 0))
            rect = alert.get_rect(center=(self.width // 2, 60))
            pygame.draw.rect(self.screen, (0, 0, 0, 150), (rect.x - 10, rect.y - 5, rect.width + 20, rect.height + 10))
            self.screen.blit(alert, rect)

    # ============================================
    # MAIN LOOP
    # ============================================

    def run(self):
        print("\n" + "=" * 60)
        print("🚦 Traffic System v6.2 - Fast Emergency + Smart Priority")
        print("=" * 60)
        print(f"Emergency Speed: {Config.EMERGENCY_SPEED} (fast)")
        print(f"Detection Distance: {Config.EMERGENCY_DETECTION_DISTANCE}px (very close)")
        print(f"Congestion Priority: Most vehicles = green light first")
        print("=" * 60)
        print("Simulation Running...\n")

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: self.ai_active = not self.ai_active
                    elif event.key == pygame.K_r: self.vehicles.clear()
                    elif event.key == pygame.K_e: self.generate_emergency_event()
                    elif event.key == pygame.K_ESCAPE: self.running = False

            self.spawn_timer += 1
            if self.spawn_timer % Config.SPAWN_INTERVAL == 0:
                self.spawn_vehicle()

            self.emergency_timer += 1
            if self.emergency_timer > self.emergency_interval:
                if len(self.emergency_vehicles) < Config.MAX_EMERGENCY_VEHICLES and random.random() < 0.3:
                    self.generate_emergency_event()
                self.emergency_timer = 0

            self.traffic_controller.update(self.vehicles, self.emergency_vehicles)
            self.update_vehicles()
            self.update_emergency_vehicles()

            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()
        print("\nSimulation stopped")


# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        import torch
        print("Starting Traffic System v6.2...")
        simulator = TrafficSimulator()
        simulator.run()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install: pip install torch numpy pygame")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()