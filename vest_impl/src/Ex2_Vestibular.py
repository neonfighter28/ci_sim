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


def main():
    # step 1: reading in the data
    in_data, Canals = read_in_data(f"{os.getcwd()}\Walking_02.txt")
    # step 2: get the orientation of the sensor and adjust the sensor data accordingly
    sensor_orientation = get_sensor_orientation(in_data.get("acc")[0])
    omegas_globaly = adjust_sensor_data(in_data.get("omega"), sensor_orientation)
    acc_globaly = adjust_sensor_data(in_data.get("acc"), sensor_orientation)
    # step 3: get the global orientation of the right horizontal scc
    r_scc_globaly = get_orientation_of_r_scc(15,Canals.get("right")[0])
    # step 4: calculate the stimulation of the cupula
    stimulation = get_stimulation(omegas_globaly, r_scc_globaly)
    # step 5: get the canal transfer function
    canal_transfer_function = get_canal_transfer_function()
    # step 6: calculate the max and min deflection of the cupula
    max_deflection, min_deflection = get_max_min_deflection(canal_transfer_function, stimulation, in_data.get("rate"))
    # step 7: calculate the max and min stimulation of the otolith
    max_otolith_stimulation, min_otolith_stimulation = get_max_min_otolith_stimulation(acc_globaly)
    # step 8: calculate the nose orientation
    nose_orientation = get_nose_orientation(omegas_globaly, in_data.get("rate"))

    # Saving the max. and min. cupular displacement in the Text File "CupularDisplacement"
    CupularDisplacement = open("CupularDisplacement.txt", "w")
    CupularDisplacement.write("Maximum Cupular Displacement :")
    CupularDisplacement.write(str(max_deflection))
    CupularDisplacement.write(";    Minimum Cupular Displacement: ")
    CupularDisplacement.write(str(min_deflection))
    CupularDisplacement.write(";")
    CupularDisplacement.close()

    # Saving the max. and min. acceleration in the Text File "Acceleration"
    Acceleration = open("Acceleration.txt", "w")
    Acceleration.write("Maximum Acceleration : ")
    Acceleration.write(str(max_otolith_stimulation))
    Acceleration.write(";   Minimum Acceleration : ")
    Acceleration.write(str(min_otolith_stimulation))
    Acceleration.write(";")
    Acceleration.close()

    # Printing the nose_orientation, which was calculated in step 8
    print(f"The resulting nose orientation is : {nose_orientation};")


if __name__ == '__main__':
    # calling the main function
    main()