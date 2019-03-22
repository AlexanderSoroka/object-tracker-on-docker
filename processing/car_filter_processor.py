"""
Filter out detection with low confidence and not-a-car.
It uses https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/ as the point of truth
for car class id.
"""

__author__ = "Alexander Soroka, soroka.a.m@gmail.com"


class CarFilterProcessor:
    def __init__(self, confidence):
        """
        Initializes filter with confidence threshold
        """
        self.__confidence_threshold = confidence

    def process_next_frame(self, frame):
        """
        Filter out detection with low confidence and not-a-car
        :param frame: frame data + list of detections, map-like object with mandatory keys: image, detections
        :return: only car detections with confidence higher than specified threshold
        """
        result = dict()
        result['image'] = frame['image']
        filtered = (frame['detections']['scores'] >= self.__confidence_threshold) \
                    & (frame['detections']['class_ids'] == 3)
        result['detections'] = {
            key: frame['detections'][key][filtered] for key in frame['detections'].keys()
        }
        return result
