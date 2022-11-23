from enum import IntEnum


class TrajectoryType(IntEnum):
    RANDOM = 0
    BENDING_1D = 1
    HALF_LEMNISCATE = 2
    FULL_LEMNISCATE = 3
    SPIRAL = 4
    FLOWER = 5
