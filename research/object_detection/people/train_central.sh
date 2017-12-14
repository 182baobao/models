#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=${DIR}/../../:${PYTHONPATH}
export PYTHONPATH=${DIR}/../../slim:${PYTHONPATH}
export PYTHONPATH=${DIR}/../../object_detection:${PYTHONPATH}

TRAIN=/home/admins/cmake/faster-rcnn

python3 ${DIR}/../train.py \
--train_dir ${TRAIN} \
--pipeline_config_path ${TRAIN}/faster_rcnn_inception_v2_people.config

python3 ${DIR}/../train.py \
--train_dir ${TRAIN}/net2 \
--pipeline_config_path ${TRAIN}/net2/faster_rcnn_inception_v2_people.config