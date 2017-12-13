#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=${DIR}/../../:${PYTHONPATH}
export PYTHONPATH=${DIR}/../../slim:${PYTHONPATH}
export PYTHONPATH=${DIR}/../../object_detection:${PYTHONPATH}

DATA=/home/admins/data/people
POSTFIX=edge
echo "creating edge patch data ..."
python3 ${DIR}/data/edge_patch.py \
--root ${DATA} \
--dataset fastercnn \
--postfix ${POSTFIX}

echo "creating train tf record .."
python3 ${DIR}/../dataset_tools/create_beer_tf_record.py \
--data_dir ${DATA} \
--set train \
--postfix ${POSTFIX} \
--output_path ${DATA} \
--label_map_path ${DATA}/people.pdtxt

echo "creating val tf record .."
python3 ${DIR}/../dataset_tools/create_beer_tf_record.py \
--data_dir ${DATA} \
--set val \
--postfix ${POSTFIX} \
--output_path ${DATA} \
--label_map_path ${DATA}/people.pdtxt