import numpy as np


def check_freq_activation(t, period):
    remainder = t % period
    if np.isclose(remainder, 0):
        return True
    if np.isclose(remainder, period):
        return True

    return False
