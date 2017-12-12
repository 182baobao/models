import time
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from functools import reduce

from utils import visualization_utils as vis_util

from beer.utils.file_io import read_file
from beer.utils.file_io import get_label_from_pd_file
from beer.data.create_lists import create_file_list
from beer.data.tools import ImageDictCropper
from beer.eval.tools import read_img_xml_as_eval_info
from beer.eval.tools import is_overlap
from beer.eval.tools import get_overlap_area
from beer.eval.predict_images import parse_args
from beer.eval.predict_images import write_predictions_result
from beer.eval.predict_images import evaluate_predictions
from beer.eval.predict_images import compute_accuracy


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


def predict_image(root, output_root, checkpoint, category_index, image_lists, score, percent):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
            start_time = time.time()
            print(time.ctime())
            for idx, paths in enumerate(image_lists):
                print('predicting {} of {} images'.format(idx, len(image_lists)))
                img_path, xml_path = paths.split('&!&')
                xml_path = xml_path if xml_path[-1] == 'l' else xml_path[:-1]
                info = read_img_xml_as_eval_info(img_path, xml_path)
                cropper = ImageDictCropper(img_path)
                info['crop_shape'] = cropper.cropped_size
                cropper.update()
                images = cropper.get_images()
                _boxes = []
                _scores = []
                _classes = []
                for key, value in images.items():
                    image_np = np.array(value[:, :, (2, 1, 0)]).astype(np.uint8)
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={
                            image_tensor: image_np_expanded
                        })
                    boxes = np.squeeze(boxes)
                    classes = np.squeeze(classes).astype(np.int32)
                    scores = np.squeeze(scores)
                    boxes, classes, scores = _convert_region_box_to_global(info, boxes, classes, scores, key)
                    _boxes += boxes
                    _classes += classes
                    _scores += scores
                _boxes, _classes, _scores = _merge_region_prediction(
                    np.array(_boxes), np.array(_scores), np.array(_classes), percent)
                _boxes = np.array(_boxes)
                _classes = np.array(_classes).astype(np.int32)
                _scores = np.array(_scores)
                image = info['image'][:, :, (2, 1, 0)]
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    _boxes,
                    _classes,
                    _scores,
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=1)
                pic = Image.fromarray(image)
                pic.save(os.path.join(output_root, '{}.jpg'.format(idx)))
                gt_num, true_pre, pre_objects = evaluate_predictions(_classes, _boxes, _scores, info, score, percent)
                with open(os.path.join(root, 'gt_pre{}-{}.txt'.format(score, percent)), 'a') as txt_file:
                    print('{} {} {}'.format(idx, gt_num, true_pre), file=txt_file)
                write_predictions_result(info, pre_objects, os.path.join(output_root, '{}.xml'.format(idx)))
                print('{} elapsed time: {:.3f}s'.format(time.ctime(),
                                                        time.time() - start_time))
                if idx > 2000:
                    break


def process():
    if args.image_list != '':
        image_lists = read_file(args.image_list)
    else:
        image_root = reduce(lambda x, y: os.path.join(x, y), args.image_path.split('.'), args.root)
        image_lists, _ = create_file_list(image_root)
    output_root = args.root if args.output == '' else args.output
    score = args.score
    percent = args.percent
    output_root = os.path.join(output_root, 'pre{}-{}'.format(score, percent))
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    num_classes = args.class_num
    category_index = get_label_from_pd_file(args.label_file, num_classes)
    predict_image(args.root, output_root, args.checkpoint,
                  category_index, image_lists, score, percent)
    compute_accuracy(os.path.join(args.root, 'gt_pre{}-{}.txt'.format(score, percent)))


if __name__ == '__main__':
    args = parse_args()
    process()