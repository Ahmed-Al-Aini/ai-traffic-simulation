import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from traffic_sim.models import Direction, LightState, Position
from traffic_sim.simulator import TrafficSimulator


def run_case(light_state: LightState, offset: float) -> None:
    simulator = TrafficSimulator()
    controller = simulator.traffic_controller
    controller.lights[Direction.NORTH]['state'] = light_state
    simulator.spawn_vehicle()
    vehicle = simulator.vehicles[-1]
    vehicle['direction'] = Direction.NORTH
    vehicle['position'] = Position(
        simulator.intersection.x - simulator.lane_width * 0.5,
        simulator.intersection.y + offset,
    )
    vehicle['speed'] = 0
    vehicle['target_speed'] = 0
    vehicle['stopped'] = True

    # The vehicle is at/just beyond the stop line while the signal is not green.
    simulator.update_vehicles()

    # Once the signal allows movement, it must not remain frozen.
    controller.lights[Direction.NORTH]['state'] = LightState.GREEN
    start_y = vehicle['position'].y
    for _ in range(30):
        simulator.update_vehicles()

    assert vehicle['position'].y < start_y, (light_state, offset, vehicle)
    assert vehicle['stopped'] is False, (light_state, offset, vehicle)
    assert vehicle['target_speed'] > 0, (light_state, offset, vehicle)


for state in (LightState.RED, LightState.YELLOW):
    for offset in (54, 56, 60, 61):
        run_case(state, offset)

print('crossing_transition_fix_ok')

import pygame
pygame.quit()
