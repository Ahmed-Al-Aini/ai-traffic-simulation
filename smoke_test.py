import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from traffic_sim.simulator import TrafficSimulator

simulator = TrafficSimulator()
for _ in range(5):
    simulator.spawn_vehicle()
    simulator.traffic_controller.update(simulator.vehicles, simulator.emergency_vehicles)
    simulator.update_vehicles()
    simulator.update_emergency_vehicles()
    simulator.draw()
print(f"smoke_ok vehicles={len(simulator.vehicles)} institutions={len(simulator.dispatcher.institutions)}")
import pygame
pygame.quit()
