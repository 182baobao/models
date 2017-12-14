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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf

from beer.classifier.tools import preprocess_image
from beer.classifier.resnet_model import imagenet_resnet_v2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', type=str, default='',
        help='The directory where the ImageNet input data is stored.')
    parser.add_argument(
        '--model_dir', type=str, default='',
        help='The directory where the model will be stored.')
    parser.add_argument(
        '--resnet_size', type=int, default=34, choices=[18, 34, 50, 101, 152, 200],
        help='The size of the ResNet model to use.')
    parser.add_argument(
        '--postfix', help='postfix to file', default='patch', type=str)
    parser.add_argument(
        '--image_size', type=int, default=192,
        help='The size of the ResNet model to use.')
    parser.add_argument(
        '--label_size', type=int, default=34,
        help='class number.')
    parser.add_argument(
        '--train_epochs', type=int, default=100,
        help='The number of epochs to use for training.')
    parser.add_argument(
        '--train_number', type=int, default=10000,
        help='The number of training dataset instance.')
    parser.add_argument(
        '--epochs_per_eval', type=int, default=1,
        help='The number of training epochs to run between evaluations.')
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for training and evaluation.')
    parser.add_argument(
        '--data_format', type=str, default='channels_last',
        choices=['channels_first', 'channels_last'],
        help='A flag to override the data format used in the model. channels_first '
             'provides a performance boost on GPU but is not always compatible '
             'with CPU. If left unspecified, the data format will be chosen '
             'automatically based on whether TensorFlow was built for CPU or GPU.')
    args = parser.parse_args()
    return args


def file_names(is_training, data_dir):
    """Return file_names for dataset."""
    if is_training:
        return [os.path.join(data_dir, 'train_{}.record'.format(FLAGS.postfix))]
    else:
        return [os.path.join(data_dir, 'val_{}.record'.format(FLAGS.postfix))]


def record_parser(value, is_training):
    """Parse an ImageNet record from `value`."""
    keys_to_features = {
        'img_raw':
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'label':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'height':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'width':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'channel':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=3),
    }

    parsed = tf.parse_single_example(value, keys_to_features)

    # image = tf.image.decode_image(tf.reshape(parsed['img_raw'], shape=[]), 3)
    image = tf.decode_raw(parsed['img_raw'], tf.uint8)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = preprocess_image(
        image=tf.reshape(image, [parsed['height'], parsed['width'], parsed['channel']]),
        output_height=FLAGS.image_size,
        output_width=FLAGS.image_size,
        is_training=is_training)

    label = tf.cast(tf.reshape(parsed['label'], shape=[]), dtype=tf.int32)

    return image, tf.one_hot(label, FLAGS.label_size + 1)


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
    """Input function which provides batches for train or eval."""
    dataset = tf.data.Dataset.from_tensor_slices(file_names(is_training, data_dir))

    if is_training:
        dataset = dataset.shuffle(buffer_size=1024)

    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(lambda value: record_parser(value, is_training),
                          num_parallel_calls=5)
    dataset = dataset.prefetch(batch_size)

    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance.
        dataset = dataset.shuffle(buffer_size=1500)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels


def resnet_model_fn(features, labels, mode, params):
    """Our model_fn for ResNet to be used with our Estimator."""
    tf.summary.image('images', features, max_outputs=6)

    network = imagenet_resnet_v2(
        params['resnet_size'], FLAGS.label_size + 1, params['data_format'])
    logits = network(
        inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    loss = cross_entropy + 0.0001 * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if 'batch_normalization' not in v.name])

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Scale the learning rate linearly with the batch size. When the batch size
        # is 256, the learning rate should be 0.1.
        initial_learning_rate = 0.1 * params['batch_size'] / 256
        batches_per_epoch = FLAGS.train_number / params['batch_size']
        global_step = tf.train.get_or_create_global_step()

        # Multiply the learning rate by 0.1 at 40, 60, and 80 epochs.
        boundaries = [
            int(batches_per_epoch * epoch) for epoch in [40, 60, 80]]
        values = [
            initial_learning_rate * decay for decay in [1, 0.1, 0.01, 1e-3]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)

        # Create a tensor named learning_rate for logging purposes.
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.9)

        # Batch norm requires update_ops to be added as a train_op dependency.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes.
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_steps=100)
    resnet_classifier = tf.estimator.Estimator(
        model_fn=resnet_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'resnet_size': FLAGS.resnet_size,
            'data_format': FLAGS.data_format,
            'batch_size': FLAGS.batch_size,
        })

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_accuracy': 'train_accuracy'
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

        print('Starting a training cycle.')
        resnet_classifier.train(
            input_fn=lambda: input_fn(
                True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
            hooks=[logging_hook])

        print('Starting to evaluate.')
        eval_results = resnet_classifier.evaluate(
            input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
        print(eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS = parse_args()
    tf.app.run()
