import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from traffic_sim.config import Config
from traffic_sim.models import Direction, LightState, Position
from traffic_sim.simulator import TrafficSimulator

# Speed must ramp up gradually after joining the main road.
simulator = TrafficSimulator()
service = simulator.dispatcher.dispatch_emergency('hospital', Position(700, 100))
assert service is not None
service.on_main_road = True
service.direction = Direction.NORTH
service.position = Position(simulator.intersection.x - simulator.lane_width * 0.5, simulator.intersection.y + 100)
service.speed = 0.0
service.target_speed = 0.0
simulator.traffic_controller.lights[Direction.NORTH]['state'] = LightState.GREEN
simulator.emergency_vehicles.append(service)
simulator.update_emergency_vehicles()
assert service.speed <= Config.EMERGENCY_ACCELERATION

# A service vehicle must follow, not overtake, a vehicle in the same lane.
normal = {
    'id': 'front',
    'position': Position(service.position.x, service.position.y - 20),
    'direction': Direction.NORTH,
    'passed_intersection': False,
}
service.speed = Config.EMERGENCY_SPEED
simulator.vehicles.append(normal)
simulator.update_emergency_vehicles()
assert service.position.y > normal['position'].y
assert service.target_speed == 0.0

# Priority must use the actual travel direction, not the side of the screen.
simulator2 = TrafficSimulator()
service2 = simulator2.dispatcher.dispatch_emergency('hospital', Position(100, 100))
assert service2 is not None
service2.on_main_road = True
service2.direction = Direction.EAST
service2.position = Position(simulator2.intersection.x - 50, simulator2.intersection.y - 17.5)
simulator2.emergency_vehicles.append(service2)
simulator2.update_emergency_vehicles()
assert simulator2.traffic_controller.emergency_direction == Direction.EAST

print('service_vehicle_fix_ok')

import pygame
pygame.quit()
