#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR/../../
export PYTHONPATH=$PYTHONPATH:$DIR/../../slim
export PYTHONPATH=$PYTHONPATH:$DIR/../../object_detection

#pretrained=$DIR/../../object_detection/data/ssd_mobilenet_v1_coco_11_06_2017
#config=$DIR/../../object_detection/samples/configs/ssd_mobilenet_beer.config
#output=$pretrained/$1

DATA=/home/admins/data/beer_data/pre


# python3 $DIR/../../object_detection/export_inference_graph.py \
# --input_type image_tensor \
# --pipeline_config_path $config \
# --trained_checkpoint_prefix $pretrained/model.ckpt- \
# --output_directory $pretrained/
# python3 prediction.py

python3 $DIR/eval_tool/main.py \
--target $DATA \
--pd_file /home/admins/cmake/ssd_mobilenet_v1_coco_11_06_2017/test/frozen_inference_graph.pb \
--root $DATA/../ceshi