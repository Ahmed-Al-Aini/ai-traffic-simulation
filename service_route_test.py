import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from traffic_sim.config import Config
from traffic_sim.models import Direction, Position
from traffic_sim.simulator import TrafficSimulator

for service_type in ('hospital', 'police', 'fire', 'civil_defense'):
    simulator = TrafficSimulator()
    vehicle = simulator.dispatcher.dispatch_emergency(service_type, Position(1100, 700))
    assert vehicle is not None
    simulator.emergency_vehicles.append(vehicle)

    reached_main_road = False
    max_exit_delta = 0.0
    for _ in range(12000):
        if vehicle.on_main_road:
            reached_main_road = True
            break
        previous = Position(vehicle.position.x, vehicle.position.y)
        simulator.update_emergency_vehicles()
        delta = vehicle.position.distance_to(previous)
        max_exit_delta = max(max_exit_delta, delta)

    assert reached_main_road, (service_type, vehicle.position, vehicle.current_route_index)
    assert vehicle.direction == vehicle.entry_direction
    assert vehicle.speed <= Config.EMERGENCY_SPEED
    assert max_exit_delta <= Config.EMERGENCY_SPEED + 1e-6, (service_type, max_exit_delta)

    if vehicle.direction in (Direction.NORTH, Direction.SOUTH):
        expected_x = simulator.get_correct_lane_x(vehicle.direction)
        assert abs(vehicle.position.x - expected_x) < 1e-6, (service_type, vehicle.entry_direction, vehicle.position.x, expected_x, vehicle.position.y)
    else:
        expected_y = simulator.get_correct_lane_y(vehicle.direction)
        assert abs(vehicle.position.y - expected_y) < 1e-6, (service_type, vehicle.entry_direction, vehicle.position.x, vehicle.position.y, expected_y)

print('service_route_speed_and_alignment_ok')

import pygame
pygame.quit()
