"""Emergency institutions, vehicles, and dispatching."""

import math
import random
import uuid
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import Config
from .models import Direction, Position, TurnDirection

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
