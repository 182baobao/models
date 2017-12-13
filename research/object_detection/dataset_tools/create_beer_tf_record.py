# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage in shell file:
    #!/usr/bin/env bash
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    export PYTHONPATH=$PYTHONPATH:$DIR/../../
    export PYTHONPATH=$PYTHONPATH:$DIR/../../slim
    export PYTHONPATH=$PYTHONPATH:$DIR/../../object_detection

    POSTFIX=""
    DATA=/home/admins/data/beer_data

    echo "generating train dataset ..."
    python $DIR/../../object_detection/dataset_tools/create_beer_tf_record.py \
    --data_dir $DATA \
    --set train \
    --postfix $POSTFIX \
    --output_path $DATA/train.record \
    --label_map_path $DIR/../../object_detection/data/beer_label_map.pbtxt
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import logging
import os
import xml.etree.ElementTree as ET
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '',
                    'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('postfix', '', 'postfix of dataset')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/beer_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val']


def dict_to_tf_example(xml_path, img_path, label_map_dict):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.
  
    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.
  
    Returns:
      example: The converted tf.Example.
  
    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    key = hashlib.sha256(encoded_jpg).hexdigest()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for obj in root.iter('object'):
        difficult = bool(int(obj.find('difficult').text))
        difficult_obj.append(int(difficult))
        xml_box = obj.find('bndbox')
        xmin.append(float(xml_box.find('xmin').text) / width)
        ymin.append(float(xml_box.find('ymin').text) / height)
        xmax.append(float(xml_box.find('xmax').text) / width)
        ymax.append(float(xml_box.find('ymax').text) / height)
        classes_text.append(obj.find('name').text.encode('utf8'))
        classes.append(label_map_dict[obj.find('name').text])

    example = tf.train.Example(
        features=tf.train.Features(feature={
            'image/height':
            dataset_util.int64_feature(height),
            'image/width':
            dataset_util.int64_feature(width),
            'image/filename':
            dataset_util.bytes_feature(
                os.path.basename(xml_path).encode('utf8')),
            'image/source_id':
            dataset_util.bytes_feature(
                os.path.basename(img_path).encode('utf8')),
            'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
            'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
            'image/object/class/text':
            dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label':
            dataset_util.int64_list_feature(classes),
            'image/object/difficult':
            dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated':
            dataset_util.int64_list_feature(truncated),
            'image/object/view':
            dataset_util.bytes_list_feature(poses),
        }))
    return example


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    data_dir = FLAGS.data_dir
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path + '{}_{}.record'.format(FLAGS.set, FLAGS.postfix))
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    examples_path = os.path.join(data_dir, FLAGS.set + '{}.txt'.format(FLAGS.postfix))
    examples_list = dataset_util.read_examples_list(examples_path)
    for idx, example in enumerate(examples_list):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples_list))
        img_path, xml_path = example.split('&!&')
        tf_example = dict_to_tf_example(xml_path, img_path, label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    tf.app.run()
