"""
Simulation of the visual System
The program lets the user select an input image and excecutes the following steps
to simulate both the retinal and the primary visual cortex acitivty.

### retinal activity
# Step 1: Find farthest corner
# Step 2: Divide distance into 10 zones
# Step 3: for each zone: Find average radius
# Step 4: for each zone: calculate eccentricity
# Step 5: for each zone: calculate RFS (rfs = 6 * eccentricity)
# Step 6: for each zone: Sigma_1 = RFS / 8
# Step 7: for each zone: Sigma_2 = 1.6 * Sigma_1
# Step 8: for each zone: calculate and apply corresponding DoG function using Sigma_1 and Sigma_2

### primary visual cortex activity
# Step 1: Using original image, apply Gabor filters in different orientations (0 - 30 - 60 - 90 - 120 - 150 deg)
# Step 2: Sum the resulting images to obtain the final image

The results are first shown on screen and then saved as "DOG_out.png" and "Gabor_out.png" respectively in the excercise folder

Authors: Cedric Koller, Elias Csuka, Leander Hemmi
v2.0.0 (11.06.23)
# Run with `python Ex3_Visual.py`
Developed and tested with Python 3.11.4
"""


# Import required packages
import PySimpleGUI as sg
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import color
import math
import sys

class MyImages:
    """Reading, writing, and filtering of images"""

    def __init__(self, in_file=None):
        """Get the image-name, -data and -size

        Images have the following methods and properties:

        Input:
        ------
        in_file : string
            Path and filename of in_file

        Properties:
        -----------
        size : tuple
            Horizontal and vertical image size
        fileName : string
            Name of original file
        data : uint8
            Image data
        focus : tuple
            Horizontal and vertical fixation point

        Methods:
        --------
        - save

        """

        if in_file == None:
            self.fileName = sg.popup_get_file("", no_window=True)
        else:
            self.fileName = in_file
        raw_data = cv.imread(self.fileName, cv.IMREAD_GRAYSCALE)

        if len(raw_data.shape) == 3:  # convert to grayscale
            raw_data = cv.cvtColor(raw_data, cv.COLOR_RGB2GRAY)


        self.size = np.shape(raw_data)
        print(self.size)

        if (
            len(raw_data.shape) == 3
        ):  # flip the image upside down, assuming it is an RGB-image
            flipped_index = range(self.size[0])
            flipped_index.sort(reverse=True)
            self.data = raw_data[flipped_index, :]
        else:
            self.data = raw_data

        # show the image, and get the focus point for the subsequent calculations
        plt.imshow(self.data, "gray")
        selected_focus = plt.ginput(1)
        plt.close()
        # ginput returns a list with one selection element, take the first
        # element in this list
        self.focus = np.rint(selected_focus)[0].tolist()
        self.focus.reverse()  # for working with the data x/y's
        print(self.focus)
    
    def save(self, out_data, name):
        """Save the resulting image to a PNG-file"""

        out_file = name + "_out.png"
        try:
            plt.imsave(out_file, out_data, cmap="gray")
            print(f"Result saved to {out_file}")
        except IOError:
            print(f"Could not save {out_file}")



### functions for retinal activity
def gaussian(x, y, sigma):
    return (1.0 / (1 * math.pi * (sigma ** 2))) * math.exp(-(1.0 / (2 * (sigma ** 2))) * (x ** 2 + y ** 2))

def DOG(x, y, sigma_1, sigma_2):
    return gaussian(x, y, sigma_1) - gaussian(x, y, sigma_2)

def make_zones_and_filters(myImg):
    """Break up the image in "numZones" different circular regions about the
    chosen focus

    Input:
    ------
    myImg : myImages-object

    Returns:
    --------
    Zones : numpy array (uint8)
        Integers with same size as self.data, indicating for each pixel to which
        "Zone" it belongs.
    Filters : numpy array (floats)
        quadratic array, containing the convolution kernel for each filter

    """

    # Set the number of zones
    numZones = 10

    # For each pixel, calculate the radius from the fixation point
    imSize = np.array(myImg.size)
    corners = np.array([[1.0, 1.0],
                        [1.0, imSize[1]],
                        [imSize[0], 1.0],
                        imSize[0:2]])
    radii = np.sqrt(np.sum((corners - myImg.focus) ** 2, axis=1))
    rMax = np.max(radii)

    X, Y = np.meshgrid(np.arange(myImg.size[0]) + 1, np.arange(myImg.size[1]) + 1)
    RadFromFocus = np.sqrt((X.T - myImg.focus[0]) ** 2 + (Y.T - myImg.focus[1]) ** 2)

    # Assign each value to a Zone, based on its distance from the "focus"
    Zones = np.floor(RadFromFocus / rMax * numZones).astype(np.uint8)
    Zones[Zones == numZones] = numZones - 1  # eliminate the few maximum radii
    # Generate numZones filters, and save them to the list "Filters"
    np.set_printoptions(threshold=sys.maxsize)
    #print(Zones)
    Filters = list()

    # ------------------- Here you have to find your own filters ------------
    # ------------------- this is just a demo! ------------
    print("Constructing DoG filters...")
    for ii in range(numZones):
        # eccentricity = average radius in a zone, in pixel
        zoneRad = ( rMax / numZones * (ii + 0.5))  
        # MY CODE
        # assume 1400px image = 30cm viewed from 60cm away
        zoneRad_in_cm = (300 / 1400) * zoneRad
        view_distance = 600 #mm
        angle = np.arctan(zoneRad_in_cm / view_distance)

        # radius of the eye is typically 125mm
        circumference = 2 * np.pi * 125
        eccentricity = circumference / 360 * angle 
        RFS = 6 * eccentricity #in arcmin
        
        RFS_in_degrees = RFS / 60
        RFS_in_pixels = int((np.tan(RFS_in_degrees) * view_distance) * (1400 / 300))
        next_largest_odd = RFS_in_pixels if (RFS_in_pixels % 2 == 1) else RFS_in_pixels + 1
        # Calculating DOG parameters based of RFS, keeping correct ratio of 1:1.6
        sigma_1 = next_largest_odd / 8
        sigma_2 = sigma_1 * 1.6

        conv_size = int(next_largest_odd) # size of convolution matrix

        # Constructing convolution matrix for calculated sigmas
        DOG_matrix = np.zeros((conv_size, conv_size))
        m_height, m_width = myImg.size
        for i in range(conv_size):
            for j in range(conv_size):
                DOG_matrix[i, j] = DOG(i, j, sigma_1, sigma_2)

        curFilter = DOG_matrix
        Filters.append(curFilter)

    return (Zones, Filters)

def apply_filters(myImg, Zones, Filters, openCV=True):
        """Applying the filters to the different image zones
        Input:
        ------
        Zones : numpy array (uint8)
            Integers with same size as self.data, indicating for each pixel to which "Zone" it belongs.
        Filters: numpy array (floats)
            quadratic array, containing the convolution kernel for each filter

        Return:
        -------
        im_out : numpy array (uint8)
            Final image
        filtered_images : list of numpy arrays (uint8)
            Same length as "Filters"
            Filtered image of each Filter
        """

        # ------------------- Here you have to apply the filters ------------

        # assuming kernel is symmetric and odd
        k_size = len(Filters[9])
    
        m_height, m_width = myImg.size
        # pad so we don't get an out of bound exception
        padded = np.pad(myImg.data, (k_size, k_size), constant_values=0)
        
        # iterates through matrix, applies kernel of correct zone, and sums
        print("Applying DoG filters...")
        im_out = []
        for i in range(k_size, m_height + k_size):
            if i%30 == 0:
                print(f"{round(i / (m_height + k_size) * 100, 0)}% done")
            for j in range(k_size, m_width + k_size):
                kernel = Filters[Zones[i - k_size, j - k_size]]
                temp_k_size = len(Filters[Zones[i - k_size, j - k_size]])
                # brightness_factor is used to brighten up all zones except the middle so they dont get too dark
                brightness_factor = 2.8 if not Zones[i- k_size, j - k_size] == 0 else 1
                im_out.append(np.sum(padded[i:temp_k_size + i, j:temp_k_size + j] * kernel) * brightness_factor)
        
        im_out = np.array(im_out).reshape((m_height, m_width))
        
        return im_out
    
def gabor_filter(myImg, angle):
    
    # constants to use for the gabor filter, determined through experimentation
    sigma  = 0.14
    theta = angle
    g_lambda = 0.25
    psi = np.pi/2
    gamma = 0.5

    sigma_x = sigma
    sigma_y = sigma/gamma
    
    # Boundingbox:
    nstds = 2
    xmax = max( abs(nstds*sigma_x * np.cos(theta)), abs(nstds*sigma_y * np.sin(theta)) )
    ymax = max( abs(nstds*sigma_x * np.sin(theta)), abs(nstds*sigma_y * np.cos(theta)) )
    
    xmax = np.ceil(max(1,xmax))
    ymax = np.ceil(max(1,ymax))
    
    xmax = 1
    ymax = 1
    xmin = -xmax
    ymin = -ymax
    
    numPts = 25
    (x,y) = np.meshgrid(np.linspace(xmin, xmax, numPts), np.linspace(ymin, ymax, numPts) ) 
    
    # Rotation
    x_theta =  x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb_values = np.array(np.exp( -0.5* (x_theta**2/sigma_x**2 + y_theta**2/sigma_y**2) ) * \
         np.cos( 2*np.pi/g_lambda*x_theta + psi ), dtype=np.float32)

    gb_values = np.array(np.exp(-0.5*(x_theta**2+y_theta**2)/sigma**2)*np.cos(2.*np.pi*x_theta/g_lambda + psi),dtype=np.float32)

    return gb_values

def main(in_file=None):
    """Simulation of a retinal ipltant. The image can be selected, and the
    fixation point is chosen interactively by the user."""

    # Select the image
    myImg = MyImages(in_file)

    # Calculate retinal activity
    Zones, Filters = make_zones_and_filters(myImg)
    filtered = apply_filters(myImg, Zones, Filters)

    # Add the results to a plot to be shown later
    fig = plt.figure(figsize=(13, 8))
    fig.add_subplot(1, 3, 1)
    plt.imshow(myImg.data, "gray")
    fig.add_subplot(1, 3, 2)
    plt.imshow(filtered, "gray")

    # Save result of retinal activity
    myImg.save(filtered, "DOG")

    # Calculating gabor filters from angles (0, 30, 60, 90, 120, 150)
    print("Constructing Gabor filters...")
    kernels = []
    for i in range(6):
        kernels.append(gabor_filter(myImg, np.pi/6 * i))
    
    # Applying the filters and saving the different outputs to an array
    print("Applying Gabor filters...")
    filtered_array = np.array([cv.filter2D(myImg.data, cv.CV_32F, kernel) for kernel in kernels],
                                   dtype=np.float32)
    
    # Sum to put together the different images
    gabor_sum = sum(filtered_array)

    # Reducing value range to the typical 0 to 255 to increase contrast.
    # This has to be done since matplotlib will otherwise try to compress a huge value range down to numbers between 0 and 1, causing the resulting image to be mostly gray.
    # 0 to 255 works because that is the format our original preprocessed image is in.
    height, width = np.shape(gabor_sum)
    for i in range(height):
        for j in range(width):
            if gabor_sum[i, j] <= 0:
                gabor_sum[i, j] = 0
            elif gabor_sum[i, j] >= 255:
                gabor_sum[i, j] = 255
    
    # Showing the results
    fig.add_subplot(1, 3, 3)
    plt.imshow(gabor_sum, "gray")
    plt.show()
    # Saving the output for the primary visual cortex activity
    myImg.save(gabor_sum, "Gabor")
    print("Done!")

if __name__ == "__main__":
     main()