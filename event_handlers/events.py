"""
Defines Event interface and some common events
"""

__author__ = "Alexander Soroka, soroka.a.m@gmail.com"


import time


class Event:
    name = ""
    timestamp = time.time()

    def __init__(self, name):
        self.name = name


class EnterFrame(Event):
    def __init__(self):
        super().__init__("Object entered frame")


class LeftFrame(Event):
    def __init__(self):
        super().__init__("Object left frame")

