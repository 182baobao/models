#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=$DIR/../../:$PYTHONPATH
export PYTHONPATH=$DIR/../../slim:$PYTHONPATH
export PYTHONPATH=$DIR/../../object_detection;$PYTHONPATH

TRAIN=/home/admins/cmake/ssd_mobilenet_v1_coco_11_06_2017

python3 $DIR/../train.py \
--train_dir $TRAIN \
--pipeline_config_path $TRAIN/ssd_mobilenet_beer.config

