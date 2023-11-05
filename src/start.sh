#!/bin/bash

git clone https://github.com/open-mmlab/mmgeneration.git && pip install -r mmgeneration/requirements.txt && pip install -v -e mmgeneration/.
python app.py