Prerequisite:
1. Setup and configure tensorflow environment
2. Get repository submodules and install MaskRCNN with command `pip install .` from MaskRCNN root dir
3. Install DeepSORT with command `pip install .` from DeepSORT root dir
4. create `workspace/pretrained` folder to keep MaskRCNN and DeepSORT pretrained models
5. Download pretrained MaskRCNN model to `workspace/pretrained/model.h5` following instructions from https://github.com/matterport/Mask_RCNN
6. Download pretrained DeepSORT model to `workspace/pretrained/mars-small128.pb` following instructions from https://github.com/nwojke/deep_sort

How to run:
1. Run docker container with RTSP server:
`sudo docker run --rm -e INPUT=/tmp/video.mp4 -v /work/object-tracking/samples/vlc-record-2019-03-05-14h51m20s-rtsp___172.16.11.89-.mp4:/tmp/video.mp4 -p 8554:8554 ullaakut/rtspatt`
2. Run server.py