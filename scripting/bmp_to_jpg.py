#!/bin/python3

from PIL import Image
import os
import sys
import numpy as np

PACKAGE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_FOLDER = os.path.join(PACKAGE_DIR, "data")
FILENAME = "dino.bmp"
FILE_PATH = os.path.join(DATA_FOLDER, FILENAME)

def bmp_to_jpg(filepath=FILE_PATH):
    filename = os.path.basename(filepath)
    file_dir = os.path.dirname(filepath)
    filename_wo_ext = os.path.splitext(filename)[0]
    file_ext = os.path.splitext(filename)[1]

    if file_ext != ".bmp":
        print(filepath + " is not a path to a bmp file!")
        return 1

    img = Image.open(filepath)
    img.save(os.path.join(file_dir, filename_wo_ext+".jpg"))
    return 0

if __name__=='__main__':
    if len(sys.argv) > 1:
        bmp_to_jpg(sys.argv[1])
    else:
        bmp_to_jpg()
