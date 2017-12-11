#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=$PYTHONPATH:$DIR/../../
export PYTHONPATH=$PYTHONPATH:$DIR/../../slim
export PYTHONPATH=$PYTHONPATH:$DIR/../../object_detection

DATA=/home/admins/data/beer_data

#python3 $DIR/eval_tool/predict_images.py \
#--image-list $DATA/pre.txt

python3 $DIR/eval_tool/predict_large_image.py \
--image-path ceshi