"""
Simple Simulation of the vestibular System
The program reads the input file "Walking_02.txt", containing measured data of acceleration, angular velocity,
quaternions and frequency. With the given data it calculates the maximum and minimum displacement of the cupula and the
maximum and minimum acceleration of the otoliths as well as the resulting orientation of the nose.
The calculated results then get saved in the files "CupularDisplacement.txt" and "Acceleration.txt". The resulted nose
orientation gets printed to the console.

Authors: Cedric Koller, Elias Csuka, Leander Hemmi
v3.1.0 (10.05.23)
# Run with `python Ex2_Vestibular.py`
Developed and tested with Python 3.8.5
"""

# Import required packages
from functions import *
import os
from tkinter.filedialog import askopenfilename
from Canals import Canals
from skinematics.sensors.xsens import XSens

def main():
    # step 1: reading in the data
    sensor = XSens(askopenfilename())
    # step 2: get the orientation of the sensor and adjust the sensor data accordingly
    sensor_orientation = get_sensor_orientation(sensor.acc[0])
    omegas_globally = adjust_sensor_data(sensor.omega, sensor_orientation)
    acc_globaly = adjust_sensor_data(sensor.acc, sensor_orientation)
    # step 3: get the global orientation of the right horizontal scc
    r_scc_globaly = get_orientation_of_r_scc(15, Canals.right[0])
    # step 4: calculate the stimulation of the cupula
    stimulation = get_stimulation(omegas_globally, r_scc_globaly)
    # step 5: get the canal transfer function
    canal_transfer_function = get_canal_transfer_function()
    # step 6: calculate the max and min deflection of the cupula
    max_deflection, min_deflection = get_max_min_deflection(canal_transfer_function, stimulation, sensor.rate)
    # step 7: calculate the max and min stimulation of the otolith
    max_otolith_stimulation, min_otolith_stimulation = get_max_min_otolith_stimulation(acc_globaly)
    # step 8: calculate the nose orientation
    nose_orientation = get_nose_orientation(omegas_globally, sensor.rate)

    # Saving the max. and min. cupular displacement in the Text File "CupularDisplacement"
    with open("CupularDisplacement.txt", "w+") as f:
        f.write(f"""
        Maximum Cupular Displacement
        {max_deflection}
        Minimum Cupular Displacement
        {min_deflection}
        """)
    # Saving the max. and min. acceleration in the Text File "Acceleration"
    with open("Acceleration.txt", "w+") as f:
        f.write(f"""
        Maximum Acceleration:
        {max_otolith_stimulation}
        Minimum Acceleration:
        {min_otolith_stimulation}""")


    # Printing the nose_orientation, which was calculated in step 8
    print(f"The resulting nose orientation is : {nose_orientation};")


if __name__ == '__main__':
    # calling the main function
    main()