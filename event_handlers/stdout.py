"""
Trivial handler that just prints events to std out
"""

__author__ = "Alexander Soroka, soroka.a.m@gmail.com"


class Handler:
    def __call__(self, events):
        print('Process {} events'.format(len(events)))
        for event in events:
            print('{}: name="{}"'.format(event.timestamp, event.name))
