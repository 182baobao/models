#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$DIR/../../:$PYTHONPATH
export PYTHONPATH=$DIR/../../slim:$PYTHONPATH
export PYTHONPATH=$DIR/../../object_detection;$PYTHONPATH

DATA=/home/admins/data/beer_data

python3 $DIR/eval_tool/predict_images.py \
--image-list $DATA/pre.txt

python3 $DIR/eval_tool/predict_large_image.py \
--image-path ceshi