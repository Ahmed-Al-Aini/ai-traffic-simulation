"""Pygame traffic simulation and rendering loop."""

import math
import random
import uuid
from typing import Dict, List, Tuple

import pygame

from .ai import TrafficAIAgent
from .config import Config
from .controller import TrafficLightController
from .emergency import EmergencyDispatcher, EmergencyVehicle
from .models import Direction, LightState, Position, TurnDirection, VehicleStatus

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
            # Urban background palette: the former grass field is now a city block.
            'urban_base': (188, 193, 188), 'lot': (214, 211, 198), 'lot_line': (155, 158, 151),
            'building_wall': (202, 176, 145), 'building_wall_alt': (177, 190, 198),
            'building_roof': (112, 80, 72), 'building_roof_alt': (75, 91, 105),
            'window': (104, 172, 190), 'window_glow': (248, 219, 139), 'door': (91, 68, 55),
            'grass': (188, 193, 188), 'stop_line': (255, 255, 255), 'intersection_zone': (100, 100, 120),
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
            # Once a vehicle reaches the crossing boundary, it must clear it
            # even if the signal changes to yellow or red.
            'committed_to_cross': False,
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
                vehicle['committed_to_cross'] = True
            elif in_square:
                vehicle['touching_intersection'] = True
                vehicle['committed_to_cross'] = True

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

            if not vehicle.get('entered_intersection') and not vehicle.get('committed_to_cross'):
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

            # A vehicle that has touched the crossing boundary is committed
            # to clearing the intersection; never stop it because of a later
            # yellow/red transition or a following-distance check.
            if (vehicle.get('committed_to_cross') or vehicle.get('entered_intersection')) and not vehicle.get('passed_intersection'):
                should_stop = False
                vehicle['target_speed'] = vehicle['max_speed']
                vehicle['stopped'] = False
            elif should_stop:
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
            arrived = vehicle.update_position(
                self.intersection,
                self.traffic_controller,
                self.vehicles + [
                    other for other in self.emergency_vehicles if other is not vehicle
                ],
            )
            if vehicle.arrived:
                self.metrics['emergency_completed'] += 1
                self.dispatcher.complete_emergency(vehicle.id)
                self.emergency_vehicles.remove(vehicle)

    # ============================================
    # RENDERING
    # ============================================

    def draw(self):
        # Draw a city environment instead of a single green background.
        self.screen.fill(self.COLORS['urban_base'])
        self._draw_urban_background()
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

    def _draw_urban_background(self):
        """Render deterministic residential/commercial blocks around the intersection."""
        ix, iy = int(self.intersection.x), int(self.intersection.y)
        road_edge = self.road_width // 2 + 12

        # Block parcels and narrow internal access streets make the empty space read as a city.
        blocks = [
            (18, 18, ix - road_edge - 18, iy - road_edge - 18),
            (ix + road_edge + 18, 18, self.width - ix - road_edge - 36, iy - road_edge - 18),
            (18, iy + road_edge + 18, ix - road_edge - 18, self.height - iy - road_edge - 36),
            (ix + road_edge + 18, iy + road_edge + 18, self.width - ix - road_edge - 36, self.height - iy - road_edge - 36),
        ]
        for bx, by, bw, bh in blocks:
            if bw <= 30 or bh <= 30:
                continue
            pygame.draw.rect(self.screen, self.COLORS['lot'], (bx, by, bw, bh), border_radius=4)
            pygame.draw.rect(self.screen, self.COLORS['lot_line'], (bx, by, bw, bh), 2, border_radius=4)
            # A small paved lane divides each urban block into properties.
            if bw > bh:
                divider_y = by + bh // 2
                pygame.draw.line(self.screen, (169, 170, 164), (bx + 8, divider_y), (bx + bw - 8, divider_y), 7)
            else:
                divider_x = bx + bw // 2
                pygame.draw.line(self.screen, (169, 170, 164), (divider_x, by + 8), (divider_x, by + bh - 8), 7)

        # Houses/buildings are positioned in the four corners and never overlap traffic lanes.
        buildings = [
            (42, 42, 108, 70, self.COLORS['building_wall'], self.COLORS['building_roof']),
            (185, 30, 92, 92, self.COLORS['building_wall_alt'], self.COLORS['building_roof_alt']),
            (self.width - 170, 38, 112, 76, self.COLORS['building_wall_alt'], self.COLORS['building_roof_alt']),
            (self.width - 315, 28, 102, 92, self.COLORS['building_wall'], self.COLORS['building_roof']),
            (45, self.height - 142, 118, 82, self.COLORS['building_wall_alt'], self.COLORS['building_roof_alt']),
            (205, self.height - 132, 92, 70, self.COLORS['building_wall'], self.COLORS['building_roof']),
            (self.width - 172, self.height - 138, 116, 84, self.COLORS['building_wall'], self.COLORS['building_roof']),
            (self.width - 318, self.height - 128, 104, 70, self.COLORS['building_wall_alt'], self.COLORS['building_roof_alt']),
        ]
        for x, y, w, h, wall, roof in buildings:
            self._draw_building(x, y, w, h, wall, roof)

        # Small trees and shrubs add scale without returning to a blank green field.
        for x, y in [(30, 155), (155, 150), (self.width - 30, 150), (self.width - 155, 150),
                     (30, self.height - 175), (155, self.height - 168),
                     (self.width - 30, self.height - 176), (self.width - 155, self.height - 166)]:
            pygame.draw.circle(self.screen, (77, 119, 75), (x, y), 9)
            pygame.draw.circle(self.screen, (105, 143, 82), (x - 4, y - 4), 5)
            pygame.draw.line(self.screen, (96, 75, 54), (x, y + 7), (x, y + 15), 3)

    def _draw_building(self, x, y, width, height, wall, roof):
        """Draw a compact building with roof, windows, entrance and facade depth."""
        shadow = (116, 121, 117)
        pygame.draw.rect(self.screen, shadow, (x + 5, y + 6, width, height), border_radius=3)
        pygame.draw.rect(self.screen, wall, (x, y, width, height), border_radius=3)
        pygame.draw.polygon(self.screen, roof, [(x - 5, y + 4), (x + width // 2, y - 15), (x + width + 5, y + 4)])
        pygame.draw.line(self.screen, (238, 226, 204), (x + 5, y + 8), (x + width - 5, y + 8), 2)
        cols = max(2, min(4, width // 30))
        rows = max(1, min(3, height // 30))
        for row in range(rows):
            for col in range(cols):
                wx = x + 12 + col * ((width - 24) // cols)
                wy = y + 17 + row * ((height - 30) // rows)
                pygame.draw.rect(self.screen, self.COLORS['window'], (wx, wy, 11, 8), border_radius=1)
                pygame.draw.line(self.screen, (222, 235, 226), (wx + 5, wy), (wx + 5, wy + 8), 1)
        pygame.draw.rect(self.screen, self.COLORS['door'], (x + width // 2 - 7, y + height - 22, 14, 22))

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
