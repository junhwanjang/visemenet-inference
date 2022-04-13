# visemenet-inference
- Inference Demo of ["VisemeNet-tensorflow"](https://github.com/yzhou359/VisemeNet_tensorflow) 
    * VisemeNet is an audio-driven animator centric speech animation driving a JALI or standard FACS-based face-rigging from input audio. 
    * The original repo is outdated and difficult to setup the environment for testing the pretrained model. This code is to provide a super-clean inference module based on the original author's repo.

## Model Inference
[Colab Demo](https://colab.research.google.com/drive/1dS4chsaFdC1D3vBaIaqIhBNaGSzXtzAc?usp=sharing)

- This code provides the simple and clean inference code without any needless ones
- It's compatible with TF 2.0 Version

### Requirements
* Tensorflow 2.x
* numpy
* scipy
* python_speech_features

### How to run inference
```python
import numpy as np
from inference import VisemeRegressor

pb_filepath = "./visemenet_frozen.pb"
wav_file_path = "./test_audio.wav"
out_txt_path = "./maya_viseme_outputs.txt"

viseme_regressor = VisemeRegressor(pb_filepath=pb_filepath)

viseme_outputs = viseme_regressor.predict_outputs(wav_file_path=wav_file_path)

np.savetxt(out_txt_path, viseme_outputs, '%.4f')
```

## How to freeze graph
- This repo does not need bazel-build for "freeze-graph" function
- Thanks to https://github.com/lighttransport/VisemeNet-infer for giving some examples.

### Requirements
* Python 3.6.x using ["pyenv"](https://github.com/pyenv/pyenv)
* Tensorflow 1.1.0

1. Setup the envs and packages
```shell
# Install Virtualenv using pyenv
pyenv install 3.6.5
pyenv virtualenv 3.6.5 visemenet-freeze
pyenv activate visemenet-freeze
```
```shell
# Install packages
pip install tensorflow==1.1.0
```

2. Clone the repo
```shell
# Clone Visemenet repo and the pretrained model
git clone https://github.com/yzhou359/VisemeNet_tensorflow.git
curl -L https://www.dropbox.com/sh/7nbqgwv0zz8pbk9/AAAghy76GVYDLqPKdANcyDuba?dl=0 > pretrained_model.zip
unzip prtrained_model.zip -d VisemeNet_tensorflow/data/ckpt/pretrain_biwi/
```

3. Freeze Graph and Save as pb
```shell
# Freeze Graph
python freeze_graph.py
```
