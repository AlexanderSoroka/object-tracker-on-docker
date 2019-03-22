"""
RTSP data source implements python iterator interface. Grab one frame on demand without caching.
Can freeze caller if request rate is higher then stream framerate
"""

__author__ = "Alexander Soroka, soroka.a.m@gmail.com"

import cv2
import time

from multiprocessing import Queue, Process


def stream_reader(uri, queue):
    stream = cv2.VideoCapture(uri)
    while stream.grab():
        if queue.empty():
            succeeded, frame = stream.retrieve()
            queue.put(frame)
#        time.sleep(0.01)


class RTSPSource:
    """
    Iterator wrapper for opencv VideoCapture
    """
    def __init__(self, uri):
        """
        Initialize new stream with specified uri
        :param uri: RSTP source uri
        """
        self.__frame_queue = Queue()
        self.__reader = Process(target=stream_reader, args=(uri, self.__frame_queue))
        self.__reader.start()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        try:
            frame = self.__frame_queue.get(block=False)
        except:
            raise StopIteration

        return frame
