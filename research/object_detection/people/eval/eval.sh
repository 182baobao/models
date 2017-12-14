#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${DIR}/../../../:${PYTHONPATH}
export PYTHONPATH=${DIR}/../../../slim:${PYTHONPATH}
export PYTHONPATH=${DIR}/../../../object_detection:${PYTHONPATH}

DATA=/home/admins/repos/trained/people-faster-rcnn
pretrained=${DATA}
config=${DATA}/faster_rcnn_inception_v2_pets.config

#python3 ${DIR}/../../../object_detection/export_inference_graph.py \
#--input_type image_tensor \
#--pipeline_config_path ${config} \
#--trained_checkpoint_prefix ${pretrained}/model.ckpt-15657 \
#--output_directory ${pretrained}/graph

python3 ${DIR}/predict_images.py \
--root ${DATA}/eval/origin \
--output-root ${DATA}/eval/predict \
--checkpoint ${DATA}/graph/frozen_inference_graph.pb \
--label-file ${DATA}/people.pdtxt