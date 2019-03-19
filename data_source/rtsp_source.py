"""
RSTP data source implements python iterator interface. Grab one frame on demand without caching.
Can freeze caller if request rate is higher then stream framerate
"""


__author__ = "Alexander Soroka, soroka.a.m@gmail.com"


import cv2


class RTSPSource:
    """
    Iterator wrapper for opencv VideoCapture
    """
    def __init__(self, uri):
        """
        Initialize new stream with specified uri
        :param uri: RSTP source uri
        """
        self.__stream = cv2.VideoCapture(uri)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        succeeded, frame = self.__stream.read()
        if succeeded:
            return frame
        else:
            raise StopIteration()
