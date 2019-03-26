"""
Performs object tracking using DeepSORT algorithm
"""

__author__ = "Alexander Soroka, soroka.a.m@gmail.com"

import numpy as np
from deep_sort import nn_matching
from deep_sort.iou_matching import iou
from deep_sort.detection import Detection
from deep_sort.track import TrackState
from deep_sort.tracker import Tracker
from deep_sort.tools.generate_detections import create_box_encoder


class DeepSORTProcessor:
    """
    Implements multiple objects tracker based on DeepSORT
    """
    def __init__(self):
        """
        Initializes DeepSORT session, i.e. object tracks and tracker internal state are persistent until object goes
        under garbage collector
        """
        self.__feature_extractor = create_box_encoder('/work/object-tracking/workspace/pretrained/mars-small128.pb')
        self.__max_cosine_distance = 0.9
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.__max_cosine_distance, None)
        self.__tracker = Tracker(metric, n_init=5, max_age=100)

    def process_next_frame(self, frame):
        """
        Track objects from specified detections
        :param frame: frame data + list of detections, map-like object with mandatory keys: image, detections
        :return: detections populated with object ids
        """
        if len(frame['detections']['rois']) == 0:
            return frame

        frame['detections']['rois'][:, 2] -= frame['detections']['rois'][:, 0]
        frame['detections']['rois'][:, 3] -= frame['detections']['rois'][:, 1]
        frame['detections']['features'] = self.__feature_extractor(frame['image'], frame['detections']['rois'])

        self.__tracker.predict()
        self.__tracker.update([
            Detection(
                frame['detections']['rois'][idx],
                frame['detections']['scores'][idx],
                frame['detections']['features'][idx]
            ) for idx, d in enumerate(frame['detections']['rois'])
        ])

        tracked_bbox = [track.to_tlwh() for track in self.__tracker.tracks if track.state == TrackState.Confirmed]
        for idx_detection, detection in enumerate(frame['detections']['rois']):
            for idx_track, track in enumerate(tracked_bbox):
                if iou(
                        tracked_bbox[idx_track],
                        np.array(frame['detections']['rois'][idx_detection], dtype=np.float).reshape(1, 4)
                )[0] >= 0.7:
                    frame['detections']['ids'][idx_detection] = self.__tracker.tracks[idx_track].track_id
        return frame
