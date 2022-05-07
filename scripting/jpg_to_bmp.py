#!/bin/python3

from PIL import Image
import os
import sys
import numpy as np

PACKAGE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_FOLDER = os.path.join(PACKAGE_DIR, "data")
FILENAME = "dino.jpg"
FILE_PATH = os.path.join(DATA_FOLDER, FILENAME)

def jpg_to_bmp(filepath=FILE_PATH):
    filename = os.path.basename(filepath)
    file_dir = os.path.dirname(filepath)
    filename_wo_ext = os.path.splitext(filename)[0]
    file_ext = os.path.splitext(filename)[1]

    if file_ext != ".jpg":
        print(filepath + " is not a path to a jpeg file!")
        return 1

    img = Image.open(filepath)
    img.save(os.path.join(file_dir, filename_wo_ext+".bmp"))
    return 0

if __name__=='__main__':
    if len(sys.argv) > 1:
        jpg_to_bmp(sys.argv[1])
    else:
        jpg_to_bmp()
