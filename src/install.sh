#!/bin/bash

mim install mmcv-full==1.7.1

cd /home/xlab-app-center/src/mmgeneration
pip install -r requirements.txt
pip install -v -e .
