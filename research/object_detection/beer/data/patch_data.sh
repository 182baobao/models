#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$DIR/../../../:$PYTHONPATH
export PYTHONPATH=$DIR/../../../slim:$PYTHONPATH
export PYTHONPATH=$DIR/../../../object_detection:$PYTHONPATH
DATA=/home/admins/data/beer_data

## create label list
:<<!
echo "creating beer label list ..."
python3 $DIR/create_lists.py \
--root $DATA \
--target $DIR/../../data
!
## extract patch dataset


POSTFIX=patch
echo "extracting image patch ..."
python3 $DIR/extract_patch.py \
--root $DATA \
--target patch \
--postfix $POSTFIX \
--label_file $DIR/../../data/beer_label.txt \
--instance 2000