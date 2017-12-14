import time
import os
import tensorflow as tf
import numpy as np
from PIL import Image

from utils import visualization_utils as vis_util

from beer.data.tools import ImageDictCropper
from beer.eval.tools import is_overlap
from beer.eval.tools import get_overlap_area
from beer.eval.predict_images import parse_args
from beer.eval.predict_images import run_detection
from beer.utils.file_io import get_label_from_pd_file
from beer.utils.file_io import read_file


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


def _merge_region_prediction(boxes, scores, classes, percent):
    idx = np.argsort(-scores)
    boxes = boxes[idx]
    scores = scores[idx]
    classes = classes[idx]
    _boxes = [boxes[0]]
    _scores = [scores[0]]
    _classes = [classes[0]]
    for box, score_, cls in zip(boxes, scores, classes):
        is_add = True
        for _box, _sc, _cls in zip(_boxes, _scores, _classes):
            if is_overlap(box, _box):
                src_area = min((_box[2] - _box[0]) * (_box[3] - _box[1]),
                               (box[2] - box[0]) * (box[3] - box[1]))
                area = get_overlap_area(_box, box)
                if (area / src_area) > percent:
                    is_add = False
                    break
        if is_add:
            _boxes.append(box)
            _scores.append(score_)
            _classes.append(cls)
    return _boxes, _classes, _scores


def _convert_region_box_to_global(info, boxes, classes, scores, index):
    src_h, src_w = info['shape']
    h, w = info['crop_shape']
    idx_x, idx_y = int(index.split('_')[0]), int(index.split('_')[1])
    _boxes = []
    _scores = []
    _classes = []
    for box, score, cls in zip(boxes, scores, classes):
        xmin = (box[1] * w + idx_x) / src_w
        ymin = (box[0] * h + idx_y) / src_h
        xmax = (box[3] * w + idx_x) / src_w
        ymax = (box[2] * h + idx_y) / src_h
        if (xmax >= 1.0) or (ymax >= 1.0):
            continue
        _boxes.append([ymin, xmin, ymax, xmax])
        _scores.append(score)
        _classes.append(cls)
    return _boxes, _classes, _scores


def predict_image(checkpoint, label_file, image_lists, score, percent):
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
    category_index = get_label_from_pd_file(label_file)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
            start_time = time.time()
            for idx, image in enumerate(image_lists):
                top_img, bottom_img = slice_image(np.array(image))
                # top image
                top_cropper = ImageDictCropper(image, (320, 320), (120, 200))
                top_cropper.update()
                top_images = top_cropper.get_images()
                _boxes = []
                _scores = []
                _classes = []
                for key, value in top_images.items():
                    image_np = value.astype(np.uint8)
                    boxes, classes, scores = run_detection(sess, detection_graph, image_np)
                    boxes, classes, scores = _convert_region_box_to_global(
                        {'shape': top_img.shape, 'crop_shape': (320, 320)}, boxes, classes, scores, key)
                    boxes, classes, scores = _convert_region_box_to_global(
                        {'shape': image.shape, 'crop_shape': top_img.shape}, boxes, classes, scores, key)
                    _boxes += boxes
                    _classes += classes
                    _scores += scores
                # bottom image
                bottom_cropper = ImageDictCropper(image, (320, 320), (120, 200))
                bottom_cropper.update()
                bottom_images = bottom_cropper.get_images()
                for key, value in bottom_images.items():
                    input_img = np.flip(value, 0)
                    image_np = input_img.astype(np.uint8)
                    boxes, classes, scores = run_detection(sess, detection_graph, image_np)
                    boxes = np.vstack((1 - boxes[:, 0], boxes[:, 1], 1 - boxes[:, 2], boxes[:, 3])).T
                    boxes, classes, scores = _convert_region_box_to_global(
                        {'shape': bottom_img.shape, 'crop_shape': (320, 320)}, boxes, classes, scores, key)
                    boxes, classes, scores = _convert_region_box_to_global(
                        {'shape': image.shape, 'crop_shape': bottom_img.shape}, boxes, classes, scores, key)
                    _boxes += boxes
                    _classes += classes
                    _scores += scores
                _boxes, _classes, _scores = _merge_region_prediction(
                    np.array(_boxes), np.array(_scores), np.array(_classes), percent)
                _boxes = np.array(_boxes)
                _classes = np.array(_classes).astype(np.int32)
                _scores = np.array(_scores)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    _boxes,
                    _classes,
                    _scores,
                    category_index,
                    use_normalized_coordinates=True,
                    min_score_thresh=score,
                    line_thickness=1)
                pic = Image.fromarray(image)
                pic.save(os.path.join(args.root, '{}.jpg'.format(idx)))
                output_images.append(image)
            time_cost = time.time() - start_time
    return output_images, time_cost


if __name__ == '__main__':
    args = parse_args()
    image_lists = read_file(args.image_list)
    predict_image(args.checkpoint, args.label_file, image_lists, 0.3, 0.75)
