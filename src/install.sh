#!/bin/bash

pip install -U openmim
mim install mmcv-full==1.7.1

cd mmgeneration
pip install -r requirements.txt
pip install -v -e .
