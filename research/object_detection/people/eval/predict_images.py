import time
import os
import tensorflow as tf
import numpy as np
import cv2

from utils import visualization_utils as vis_util

from beer.data.tools import ImageDictCropper
from beer.data.create_lists import create_file_list
from beer.eval.predict_images import parse_args
from beer.eval.predict_images import run_detection
from beer.eval.predict_large_image import convert_region_box_to_global
from beer.eval.predict_large_image import merge_region_prediction
from beer.utils.file_io import get_label_from_pd_file


def slice_image(image, axis=0, wrap_pixel=128, wrap_ratio=0, slice_seat=0.5):
    seat = int(image.shape[axis] * slice_seat)
    if 0 < wrap_ratio < 1:
        wrap_pixel = int(image.shape[axis] * wrap_ratio)
    if axis == 0:
        former = image[:(seat + wrap_pixel // 2), :, :]
        later = image[(seat + wrap_pixel // 2):, :, :]
    else:
        former = image[:, (seat + wrap_pixel // 2), :]
        later = image[:, (seat + wrap_pixel // 2):, :]
    return former, later


def predict_image(checkpoint, label_file, image_list, score, percent):
    output_images = []

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    category_index = get_label_from_pd_file(label_file, 1)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
            start_time = time.time()
            for idx, image in enumerate(image_list):
                img = cv2.imread(image)[:, :, (2, 1, 0)] if isinstance(image, str) else image
                top_img, bottom_img = slice_image(img)
                # top image
                top_cropper = ImageDictCropper(top_img, (320, 320), (120, 200))
                top_cropper.update()
                top_images = top_cropper.get_images()
                _boxes = []
                _scores = []
                _classes = []
                for key, value in top_images.items():
                    image_np = value.astype(np.uint8)
                    boxes, classes, scores = run_detection(sess, detection_graph, image_np)
                    boxes, classes, scores = convert_region_box_to_global(
                        {'shape': top_img.shape, 'crop_shape': (320, 320)}, boxes, classes, scores, key)
                    boxes, classes, scores = convert_region_box_to_global(
                        {'shape': img.shape, 'crop_shape': top_img.shape}, boxes, classes, scores, key)
                    _boxes += boxes
                    _classes += classes
                    _scores += scores
                # bottom image
                bottom_cropper = ImageDictCropper(bottom_img, (320, 320), (120, 200))
                bottom_cropper.update()
                bottom_images = bottom_cropper.get_images()
                for key, value in bottom_images.items():
                    image_np = np.flip(value, 0).astype(np.uint8)
                    boxes, classes, scores = run_detection(sess, detection_graph, image_np)
                    boxes = np.vstack((1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3])).T
                    boxes, classes, scores = convert_region_box_to_global(
                        {'shape': bottom_img.shape, 'crop_shape': (320, 320)}, boxes, classes, scores, key)
                    boxes, classes, scores = convert_region_box_to_global(
                        {'shape': img.shape, 'crop_shape': bottom_img.shape}, boxes, classes, scores, key)
                    _boxes += boxes
                    _classes += classes
                    _scores += scores
                _boxes, _classes, _scores = merge_region_prediction(
                    np.array(_boxes), np.array(_scores), np.array(_classes), percent)
                _boxes = np.array(_boxes)
                _classes = np.array(_classes).astype(np.int32)
                _scores = np.array(_scores)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    img,
                    _boxes,
                    _classes,
                    _scores,
                    category_index,
                    use_normalized_coordinates=True,
                    min_score_thresh=score,
                    line_thickness=1)
                cv2.imwrite(os.path.join(args.output, '{}.jpg'.format(idx)), img)
                output_images.append(image)
            time_cost = time.time() - start_time
    return output_images, time_cost


if __name__ == '__main__':
    args = parse_args()
    image_lists, _ = create_file_list(args.root)
    predict_image(args.checkpoint, args.label_file, image_lists, 0.3, 0.75)
