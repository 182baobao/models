import os
import logging
import tensorflow as tf
from PIL import Image

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '',
                    'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('postfix', '', 'postfix of dataset')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_size', 112, 'size of input image')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val']


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    data_dir = FLAGS.data_dir
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    examples_path = os.path.join(data_dir, FLAGS.set + '{}.txt'.format(FLAGS.postfix))
    examples_list = dataset_util.read_examples_list(examples_path)
    for idx, example in enumerate(examples_list):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples_list))
        img_path, label = example.split('&!&')
        img = Image.open(img_path)
        img = img.resize((FLAGS.image_size, FLAGS.image_size))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
        writer.write(example.SerializeToString())
    writer.close()
    

if __name__ == '__main__':
    tf.app.run()
