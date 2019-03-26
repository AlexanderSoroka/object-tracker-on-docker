"""
Implementation of single store of object tracks. Keep both actual object tracks and tracks for object which left frame
"""

__author__ = "Alexander Soroka, soroka.a.m@gmail.com"


import numpy as np


class HistoricalTracksProcessor:
    """
    Internal track format is: dict={
        object-id: {
            "frames": [t1, t2, t3....],
            "points": [[y,x], [y,x], [y,x]...]
        }
    }
    """
    def __init__(self):
        self.__frame_number = 0
        self.__tracks = {}

    def process_next_frame(self, frame):
        for idx, track_id in enumerate(frame['detections']['ids']):
            if track_id == 0:
                continue

            roi = frame['detections']['rois'][idx]
            if track_id not in self.__tracks.keys():
                self.__tracks[track_id] = {
                    'frames': [],
                    'points': []
                }
            self.__tracks[track_id]['frames'].append(self.__frame_number)
            self.__tracks[track_id]['points'].append([roi[1] + int(roi[3]/2), roi[0] + int(roi[2]/2)])

        self.__frame_number += 1
        return self.__tracks
