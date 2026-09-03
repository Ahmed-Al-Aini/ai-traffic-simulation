"""Traffic-light state machine and priority logic."""

from typing import Dict, List, Optional

from .config import Config
from .emergency import EmergencyVehicle
from .models import Direction, LightState, Position

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
        self.congestion_check_interval = 30  # Check congestion every 30 frames

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
