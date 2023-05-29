""" Demo of how to apply radius-dependent filter to an image. """

# Author: Thomas Haslwanter
# Date:   May 2023


import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2
import PySimpleGUI as sg
from skimage import color


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
        - apply_filters
        - save

        """

        if in_file == None:
            self.fileName = sg.popup_get_file("", no_window=True)
        else:
            self.fileName = in_file
        raw_data = plt.imread(self.fileName)

        if len(raw_data.shape) == 3:  # convert to grayscale
            raw_data = color.rgb2gray(raw_data)

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


    def apply_filters(self, Zones, Filters, openCV=True):
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

        return im_out

    def save(self, out_data):
        """Save the resulting image to a PNG-file"""

        in_stem = self.fileName.find(".")
        out_file = self.fileName[:in_stem] + "_out.png"
        try:
            plt.imsave(out_file, out_data, cmap="gray")
            print(f"Result saved to {out_file}")
        except IOError:
            print(f"Could not save {out_file}")


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
    Filters = list()

    # ------------------- Here you have to find your own filters ------------
    # ------------------- this is just a demo! ------------
    for ii in range(numZones):
        # eccentricity = average radius in a zone, in pixel
        zoneRad = ( rMax / numZones * (ii + 0.5))  

        #  the "1/4" is arbitrary, just used to give a proper visual example
        filter_length = np.rint( zoneRad / 4)  

        # for Reasons of memory efficiency, limit filter size
        filter_length = np.int32( min(81, filter_length))

        # normalize the filter
        curFilter = ( np.ones((filter_length, filter_length)) / filter_length ** 2) 

        Filters.append(curFilter)
        print("filter_length %d: %g" % (ii, filter_length))

    return (Zones, Filters)


def main(in_file=None):
    """Simulation of a retinal ipltant. The image can be selected, and the
    fixation point is chosen interactively by the user."""

    # Select the image, and perform the calculations
    myImg = MyImages(in_file)
    Zones, Filters = make_zones_and_filters(myImg)
    filtered = myImg.apply_filters(Zones, Filters)

    # Show the results
    plt.imshow(filtered, "gray")
    plt.show()
    myImg.save(filtered)
    print("Done!")


if __name__ == "__main__":
     main()
