import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from traffic_sim.config import Config
from traffic_sim.models import Direction, LightState, Position
from traffic_sim.simulator import TrafficSimulator

simulator = TrafficSimulator()
simulator.traffic_controller.lights[Direction.NORTH]['state'] = LightState.RED
simulator.spawn_vehicle()
vehicle = simulator.vehicles[-1]
vehicle['direction'] = Direction.NORTH
vehicle['position'] = Position(simulator.intersection.x - simulator.lane_width * 0.5, simulator.intersection.y + simulator.half_width + 9)
vehicle['speed'] = 0
vehicle['target_speed'] = 0
vehicle['stopped'] = True

simulator.update_vehicles()
assert vehicle['committed_to_cross'] is True
assert vehicle['target_speed'] == vehicle['max_speed']
assert vehicle['stopped'] is False

for _ in range(20):
    simulator.update_vehicles()

assert vehicle['passed_intersection'] or vehicle['entered_intersection']
print('crossing_fix_ok')

import pygame
pygame.quit()
