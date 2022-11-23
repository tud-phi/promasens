from enum import IntEnum


class JointNNMode(IntEnum):
    """
    Enum for the joint neural network mode.
    """
    ALL = 0
    EACH_SEGMENT = 1
    EACH_SENSOR = 4
