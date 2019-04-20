"""
Implementation of license plate recognizer based on OpenALPR
"""

__author__ = "Alexander Soroka, soroka.a.m@gmail.com"


import cv2
from openalpr import Alpr


class OpenALPRProcessor:
    def __init__(self, config, share_data):
        self.__alpr = Alpr("eu-by", config, share_data)
        self.__counter = 0

    def process_next_frame(self, frame):
        self.__counter += self.__counter
        frame['detections']['plates'] = []
        for idx, roi in enumerate(frame['detections']['rois']):
            results = self.__alpr.recognize_ndarray(frame['image'][roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3]])
            if len(results['results']):
                frame['detections']['plates'].append(results['results'])
        return frame
