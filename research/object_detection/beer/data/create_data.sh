#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$DIR/../../:$PYTHONPATH
export PYTHONPATH=$DIR/../../slim:$PYTHONPATH
export PYTHONPATH=$DIR/../../object_detection;$PYTHONPATH

## object detection dataset

POSTFIX=300
DATA=/home/admins/data/beer_data
echo "cropping image ..."
python3 $DIR/data/main.py \
--root $DATA \
--target crop \
--postfix $POSTFIX

echo "generating train dataset ..."
python3 $DIR/../../object_detection/dataset_tools/create_beer_tf_record.py \
--data_dir $DATA \
--set train \
--postfix $POSTFIX \
--output_path $DATA/train.record \
--label_map_path $DIR/../../object_detection/data/beer_label_map.pbtxt
echo "generating val dataset ..."
python3 $DIR/../../object_detection/dataset_tools/create_beer_tf_record.py \
--data_dir $DATA \
--set val \
--postfix $POSTFIX \
--output_path $DATA/val.record \
--label_map_path $DIR/../../object_detection/data/beer_label_map.pbtxt