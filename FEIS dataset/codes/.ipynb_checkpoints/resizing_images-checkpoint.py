"""
Objective: Resizing 389x1089 spectrogram images to 32x32
Usage: Provide source path of folder in which images are kept for resizing and destination folder to save the resized images. (width, height)
Author: Vinay Visanji Faria
"""

import cv2
import os

source = "C:\\Users\\vinay\\Downloads\\FEIS_v1_1\\spectrogram_train_test_split\\train\\goose"
destination = "C:\\Users\\vinay\\Downloads\\FEIS_v1_1\\spectrogram_train_test_split_resized\\train\\goose\\"
folder=os.listdir(source)
for pic in folder:
    picture=cv2.imread(source +'/'+pic)
    picture=cv2.resize(picture, (32,32), interpolation= cv2.INTER_AREA)
    file_saving_location = destination + pic
    cv2.imwrite(file_saving_location,picture)