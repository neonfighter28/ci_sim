import numpy as np
import pandas as pd
from skinematics import vector
from skinematics import quat
from skinematics import rotmat
from skinematics.sensors.xsens import XSens
from scipy import signal
from scipy.constants import g

def get_sensor_orientation(acc_data):
    """
    Calculates the orientation of the sensor with respect to a global, space-fixed coordinate system.
    """
    g_vector = np.array([0, g, 0])
    remaining_rotation = vector.q_shortest_rotation(acc_data, g_vector)
    turn_ninty_x = vector.q_shortest_rotation(np.array([0, 1, 0]), np.array([0, 0, 1]))
    sensor_orientation = quat.q_mult(turn_ninty_x, remaining_rotation)
    return sensor_orientation


def get_orientation_of_r_scc(deviation_reid_head, r_scc_relative_to_reid):
    """
    Calculates the orientation of the right horizontal semicircular canal (scc) with respect to a global, space-fixed
    coordinate system by rotating the coordinates around the y-axis by 15 degrees.
    """
    q = rotmat.convert(rotmat.R("y", -deviation_reid_head))
    r_scc_globaly = vector.rotate_vector(r_scc_relative_to_reid, q)
    return r_scc_globaly


def adjust_sensor_data(sensor_data, sensor_orientation):
    """
    Adjusts the sensor data to align it with the global coordinate system by rotating it the amount of the
    sensor orientation.
    """
    adjusted_data = vector.rotate_vector(sensor_data, sensor_orientation)
    return adjusted_data


def get_stimulation(omegas, r_ssc_globaly):
    """
    Calculate the stimulation of the right horizontal semicircular canal.
    """
    stimulation = omegas @ r_ssc_globaly
    return stimulation


def get_canal_transfer_function():
    """
    Calculates the transfer function of the semicircular canals represented as a linear time invariant system.
    """
    # time-constant for the low-pass filter (sec)
    T1 = 0.01
    # time-constant for the high-pass filter (sec)
    T2 = 5
    num = [T1*T2, 0]
    den = [T1*T2, T1+T2, 1]
    # Find the bode-plot, which characterizes the system dynamics
    canal_transfer_function = signal.lti(num, den)
    return canal_transfer_function


def get_max_min_deflection(canal_transfer_function, stimulation, frequency):
    """
    Calculates the maximum and minimum deflection of the semicircular canals.
    """
    # radius of the semicircular canals (mm)
    scc_radius = 3.2
    # time axis
    t = np.arange(len(stimulation)) / frequency
    _, output, _ = signal.lsim(canal_transfer_function, stimulation, t)
    deflection = output * scc_radius
    # calculate minimum and maximum deflection
    deflection_max = np.array(np.max(deflection))
    deflection_min = np.array(np.min(deflection))
    return deflection_max, deflection_min


def get_max_min_otolith_stimulation(acc_data):
    """
    Calculates the maximum and minimum stimulation of the otolithic organ based on the acceleration data.
    """
    otolith_stimulation = np.array([acc[1] for acc in acc_data])
    max_otolithic_stimulation = np.max(otolith_stimulation)
    min_otolith_stimulation = np.min(otolith_stimulation)
    return max_otolithic_stimulation, min_otolith_stimulation


def get_nose_orientation(omegas, frequency):
    """
    Calculates the orientation of the nose.
    """
    quaternions = quat.calc_quat(omegas, [0, 0, 0], rate=frequency, CStype="bf")
    nose = [1, 0, 0]
    nose = vector.rotate_vector(nose, quaternions[-1])
    return nose
