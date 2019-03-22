#!python

"""
Implementation of single source object tracker based on SOTA deeplearning models
"""

__author__ = "Alexander Soroka, soroka.a.m@gmail.com"


import argparse
import json

from data_source.rtsp_source import RTSPSource
from event_handlers.event_handler_factory import EventHandlerFactory
from event_handlers.timer_event import TimerEvent
from processing.event_generator import EventGenerator
from processing.historical_tracks_processor import HistoricalTracksProcessor
from processing.mask_rcnn_processor import MaskRCNNProcessor
from processing.deep_sort_processor import DeepSORTProcessor
from processing.car_filter_processor import CarFilterProcessor


class Server:
    """
    Server implementation
    """
    def __init__(self, source, handlers):
        self.__counter = 0
        self.__source = source
        self.__handlers = handlers
        self.__car_filter = CarFilterProcessor(confidence=0.7)
        self.__processor = MaskRCNNProcessor()
        self.__tracker = DeepSORTProcessor()
        self.__historical_tracks = HistoricalTracksProcessor()
        self.__event_generator = EventGenerator()
        self.__visualizer = None

    def run(self):
        for frame in self.__source:
            events = self.__process(frame)
            self.__handle_events(events)
            self.__counter = self.__counter + 1

    def __process(self, frame):
        timer = TimerEvent()
        frame = {'image': frame}
        frame = self.__processor.process_next_frame(frame)
        proposals = self.__car_filter.process_next_frame(frame)
        proposals = self.__tracker.process_next_frame(proposals)
        tracks = self.__historical_tracks.process_next_frame(proposals)
        events = self.__event_generator.process_next_frame(proposals)
        events.append(timer.evaluate())
        return events

    def __handle_events(self, events):
        for handler in self.__handlers:
            handler(events)


def load_config(config_filename):
    """
    Load application config from specified json file
    :return:
    """
    with open(config_filename, 'r') as config_file:
        return json.load(config_file)


def verify_config(config):
    """
    Check config and raise exception if any mandatory option is missed
    :return:
    """
    if 'source' not in config.keys():
        raise Exception('Source section missed')

    if 'uri' not in config['source'].keys():
        raise Exception('Missed uri for specified source')

    if 'event_handlers' not in config.keys():
        raise Exception('Event handlers section missed')

    if len(config['event_handlers']) < 1:
        raise Exception('At least one event handler must be specified')


def main():
    """
    Application entry point
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='filepath to json config', default='config.json')
    options = parser.parse_args()

    config = load_config(options.config)
    source = RTSPSource(config['source']['uri'])
    event_handlers = [
        EventHandlerFactory.create(handler['module'], handler['name']) for handler in config['event_handlers']
    ]

    server = Server(source, event_handlers)
    server.run()


if __name__ == "__main__":
    main()