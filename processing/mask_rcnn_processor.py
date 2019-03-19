"""
Implement object detection and tracking
"""

__author__ = "Alexander Soroka, soroka.a.m@gmail.com"

from mrcnn.config import Config
from mrcnn import model as modellib, utils


class MaskRCNNConfig(Config):
    """
    Overload default config and add our specific settings. NB: all these self.* fields must be specified prior calling
    super constructor as it (constructor) calculates some dynamic settings
    """

    def __init__(self):
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 1
        self.BATCH_SIZE = 1
        self.NUM_CLASSES = 81
        self.NAME = "object tracking server"
        self.IMAGE_MAX_DIM = 512
        super().__init__()

    @staticmethod
    def build():
        return MaskRCNNConfig()


class MaskRCNNProcessor:
    """
    Implements calling MaskRCNN model inference for specified frame. Internal TF session persists while object is not
    garbage collected
    """

    def __init__(self):
        """
        TODO: make model weight and config paths configurable
        """
        self.__model = modellib.MaskRCNN(mode="inference", model_dir='workspace', config=MaskRCNNConfig.build())
        self.__model.load_weights('/work/object-tracking/workspace/pretrained/model.h5', by_name=True)

    def process_next_frame(self, frame):
        """
        Processes one frame and return all detections without any post-proccessing or filtering
        :param frame: numpy array with frame data of shape (h,w,ch)
        :return: array-like detections list
        """
        frame['detections'] = self.__model.detect([frame['image']], verbose=0)[0]
        return frame
