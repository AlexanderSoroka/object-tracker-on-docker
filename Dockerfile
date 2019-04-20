FROM tensorflow/tensorflow:1.13.1-gpu

RUN mkdir -p /opt && cd /opt
RUN git clone git@github.com:AlexanderSoroka/object-tracker-on-docker.git
