import numpy as np


def calculate_sound_speed_gradient(depths, sound_speeds):
    """
    Calculate the vertical change in sound speed (dc/dz).

    Parameters:
        depths: List of depth values.
        sound_speeds: List of sound speed values.

    Returns:
        NumPy array containing dc/dz.
    """
    depths = np.asarray(depths, dtype=float)
    sound_speeds = np.asarray(sound_speeds, dtype=float)

    if len(depths) != len(sound_speeds):
        raise ValueError("depths and sound_speeds must have the same length")

    if len(depths) < 2:
        raise ValueError("At least two data points are required")

    return np.gradient(sound_speeds, depths)


def detect_sound_channel(depths, sound_speeds):
    """
    Detect the Deep Sound Channel.

    The Deep Sound Channel is identified by a local minimum
    in sound speed, where the gradient changes from negative
    to positive.

    Returns:
        Depth of the detected sound channel, or None if not found.
    """
    depths = np.asarray(depths, dtype=float)
    sound_speeds = np.asarray(sound_speeds, dtype=float)

    if len(depths) != len(sound_speeds):
        raise ValueError("depths and sound_speeds must have the same length")

    if len(depths) < 3:
        return None

    gradient = np.gradient(sound_speeds, depths)

    for i in range(1, len(gradient)):
        if gradient[i - 1] < 0 and gradient[i] >= 0:
            return depths[i]

    return None


def detect_thermocline(depths, temperatures, threshold=-0.01):
    """
    Detect layers where the temperature gradient is steeply negative.

    Parameters:
        depths: List of depth values.
        temperatures: List of temperature values.
        threshold: Maximum gradient considered steeply negative.

    Returns:
        NumPy array containing the depths belonging to the thermocline.
    """
    depths = np.asarray(depths, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)

    if len(depths) != len(temperatures):
        raise ValueError("depths and temperatures must have the same length")

    if len(depths) < 2:
        return np.array([])

    temperature_gradient = np.gradient(temperatures, depths)

    return depths[temperature_gradient < threshold]