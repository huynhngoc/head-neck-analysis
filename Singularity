Bootstrap: docker
From: tensorflow/tensorflow:2.0.0-gpu-py3
Stage: build

%post
    apt-mark hold tensorflow
    apt update -y
    apt upgrade -y
    pip install ipython
    pip install deoxys
    pip install comet-ml
    pip install scikit-image
    pip install scikit-learn
    pip install mypy
    pip install nptyping

%environment
    export KERAS_MODE=TENSORFLOW
