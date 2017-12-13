#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$DIR/../../../:$PYTHONPATH
export PYTHONPATH=$DIR/../../../slim:$PYTHONPATH
export PYTHONPATH=$DIR/../../../object_detection:$PYTHONPATH
DATA=/home/admins/data/beer_data
POSTFIX=patch
## create label list
:<<!
echo "creating beer label list ..."
python3 $DIR/create_lists.py \
--root $DATA \
--target $DIR/../../data

## extract patch dataset

echo "extracting image patch ..."
python3 $DIR/extract_patch.py \
--root $DATA \
--dataset patch \
--postfix $POSTFIX \
--label_file $DIR/../../data/beer_label.txt \
--instance 2000
!

## create record data from patch images
#echo "creating patch train data ..."
#python3 $DIR/../../dataset_tools/create_classification_tf_record.py \
#--data_dir $DATA \
#--set train \
#--postfix $POSTFIX \
#--output_path $DATA

echo "creating patch val data ..."
python3 $DIR/../../dataset_tools/create_classification_tf_record.py \
--data_dir $DATA \
--set val \
--postfix $POSTFIX \
--output_path $DATA