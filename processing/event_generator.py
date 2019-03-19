"""
Generates event based on input data
"""

__author__ = "Alexander Soroka, soroka.a.m@gmail.com"


from event_handlers.events import EnterFrame, LeftFrame


class EventGenerator:
    def __init__(self):
        self.__objects = dict()

    def process_next_frame(self, frame):
        events = list()
        for idx, track_id in enumerate(frame['detections']['ids']):
            if track_id not in self.__objects.keys():
                self.__objects[track_id] = {
                    'bbox_tlwh': frame['detections']['rois'][idx],
                    'score': frame['detections']['scores'][idx],
                }
                events.append(EnterFrame())

        left = [id for id in self.__objects.keys() if id not in frame['detections']['ids']]
        for id in left:
            del self.__objects[id]
            events.append(LeftFrame())

        return events
