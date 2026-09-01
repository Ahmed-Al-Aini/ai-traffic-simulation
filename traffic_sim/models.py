"""Shared enums and data models."""

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

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
