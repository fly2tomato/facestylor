#!/bin/bash

pip install -U openmim
mim install mmcv-full==1.7.1

git clone https://github.com/open-mmlab/mmgeneration.git
cd mmgeneration
pip install -r requirements.txt
pip install -v -e .

pip install mmcls
