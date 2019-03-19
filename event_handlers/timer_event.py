"""
Simple vent that calculates the elapsed time between creation and evaluation
"""

__author__ = "Alexander Soroka, soroka.a.m@gmail.com"


import time
import event_handlers.events


class TimerEvent(event_handlers.events.Event):
    def __init__(self):
        self.__start = time.time()
        self.name = 'Not-evaluated timer event'
        self.value = 0

    def evaluate(self):
        self.value = time.time() - self.__start
        self.name = 'Elapsed time: {}'.format(self.value)
        return self

