from tkinter.filedialog import askopenfilename
import cv2 as cv
import matplotlib.pyplot as plt

class Image:
    def __init__(self, path):
        if not path:
            path = askopenfilename()
        self.image = cv.imread(path)

    def sel_fixation_pt(image):
        plt.imshow(cv.cvtColor(self.image, cv.COLOR_BGR2RGB))
        plt.title = ("Select fixation point")
        plt.axis("off")
        fix_point = plt.ginput(n = 1, timeout=30)
        return fix_point[0]

def readfile():
    path = askopenfilename()
    return cv.imread(path)

def sel_fixation_pt(image):
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title = ("Select fixation point")
    plt.axis("off")
    fix_point = plt.ginput(n = 1, timeout=30)
    return fix_point[0]
image = readfile()
fixation_point = sel_fixation_pt(image)
print(f"{fixation_point = }")