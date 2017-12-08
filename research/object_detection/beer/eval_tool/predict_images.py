import time
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import argparse
from functools import reduce

from utils import visualization_utils as vis_util

from beer.crop_tools.create_lists import create_file_list
from beer.eval_tool.tools import read_img_xml_as_eval_info
from beer.eval_tool.tools import is_overlap
from beer.eval_tool.tools import get_overlap_area
from beer.eval_tool.tools import get_label_from_pd_file
from beer.file_tool.txt_file import read_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='predictions parameters')
    parser.add_argument(
        '--root',
        dest='root',
        help='image to use',
        default='/home/admins/data/beer_data',
        type=str)
    parser.add_argument(
        '--image-path',
        dest='image_path',
        help='path of image to use, split by .',
        default='',
        type=str)
    parser.add_argument(
        '--checkpoint',
        dest='checkpoint',
        help='path to checkpoint',
        default='/home/admins/cmake/ssd_mobilenet_v1_coco_11_06_2017/test/frozen_inference_graph.pb',
        type=str)
    parser.add_argument(
        '--output-root',
        dest='output',
        help='output root',
        default='',
        type=str)
    parser.add_argument(
        '--score',
        dest='score',
        help='score threshold',
        default=0.15,
        type=float)
    parser.add_argument(
        '--percent',
        dest='percent',
        help='area percent threshold',
        default=0.75,
        type=float)
    parser.add_argument(
        '--image-list',
        dest='image_lists',
        help='path of image to use',
        default='',
        type=str)
    parser.add_argument(
        '--class-num',
        dest='class_num',
        help='number of classes',
        default=9,
        type=int)
    parser.add_argument(
        '--label-file',
        dest='label_file',
        help='path of image to use',
        default='/home/admins/cmake/models/research/object_detection/data/beer_label_map.pbtxt',
        type=str)
    args = parser.parse_args()
    return args


def evaluate_predictions(_classes, _boxes, _scores, _info, _score, _percent):
    objects_ = _info['objects']
    _pre_objects = []
    _gt_num = len(objects_)
    _true_pre = 0
    for cls, box, score in zip(_classes, _boxes, _scores):
        if score > _score:
            for _ob in objects_:
                if not is_overlap([box[1], box[0], box[3], box[2]], _ob[1:]):
                    src_area = min((box[2] - box[0]) * (box[3] - box[1]), (_ob[3] - _ob[1]) * (_ob[4] - _ob[2]))
                    area = get_overlap_area([box[1], box[0], box[3], box[2]], _ob[1:])
                    if (area / src_area > _percent) and (cls == (_ob[0] + 1)):
                        _true_pre += 1
                        _pre_objects.append([cls, *box, score])
                    break
    return _gt_num, _true_pre, _pre_objects


def write_predictions_result(_info, _pre_objects, _file_name):
    def _add_element(_root, _name, _value):
        sub_element = ET.SubElement(_root, _name)
        sub_element.text = _value

    root = ET.Element('annotation')
    size = ET.SubElement(root, 'size')
    shape = _info['shape']
    _add_element(size, 'height', str(shape[0]))
    _add_element(size, 'width', str(shape[1]))
    _add_element(size, 'depth', '3')
    origin = ET.SubElement(root, 'origin')
    objects_ = _info['objects']
    class_names = [
        'Harbin Wheat 330ML Can', 'Budweiser Beer 330ML Can',
        'Budweiser 600ML Bottle', 'Budweiser Beer 500ML Can', 'budweiser15',
        'budweiser31', 'budweiser30', 'harbin26',
        'budweiser26'
    ]
    for ob in objects_:
        ob_xml = ET.SubElement(origin, 'object')
        _add_element(ob_xml, 'name', class_names[ob[0]])
        _add_element(ob_xml, 'difficult', '0')
        bndbox = ET.SubElement(ob_xml, 'bndbox')
        _add_element(bndbox, 'xmin', str(int(ob[1] * shape[1])))
        _add_element(bndbox, 'ymin', str(int(ob[2] * shape[0])))
        _add_element(bndbox, 'xmax', str(int(ob[3] * shape[1])))
        _add_element(bndbox, 'ymax', str(int(ob[4] * shape[0])))

    prediction = ET.SubElement(root, 'prediction')
    print(len(_pre_objects))
    for ob in _pre_objects:
        ob_xml = ET.SubElement(prediction, 'object')
        _add_element(ob_xml, 'name', class_names[ob[0] - 1])
        _add_element(ob_xml, 'score', str(ob[-1]))
        _add_element(ob_xml, 'difficult', '0')
        bndbox = ET.SubElement(ob_xml, 'bndbox')
        _add_element(bndbox, 'xmin', str(int(ob[2] * shape[1])))
        _add_element(bndbox, 'ymin', str(int(ob[1] * shape[0])))
        _add_element(bndbox, 'xmax', str(int(ob[4] * shape[1])))
        _add_element(bndbox, 'ymax', str(int(ob[3] * shape[0])))
    tree = ET.ElementTree(root)
    tree.write(_file_name)


def compute_accuracy(result_file):
    result_list = read_file(result_file)
    total = 0
    true_pre = 0
    for result in result_list:
        idx, _gt_num, _pre = result.split(' ')
        total += int(_gt_num)
        true_pre += int(_pre)
    print(total)
    print(true_pre)
    print(true_pre / total)


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
                image = Image.open(img_path)
                info = read_img_xml_as_eval_info(img_path, xml_path[:-1])
                image_np = np.array(image).astype(np.uint8)
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
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=5)
                pic = Image.fromarray(image_np)
                pic.save(os.path.join(output_root, '{}.jpg'.format(idx)))
                gt_num, true_pre, pre_objects = evaluate_predictions(classes, boxes, scores, info)
                with open(os.path.join(root, 'gt_pre{}-{}.txt'.format(score, percent)), 'a') as txt_file:
                    print('{} {} {}'.format(idx, gt_num, true_pre), file=txt_file)
                write_predictions_result(info, pre_objects, os.path.join(output_root, '{}.xml'.format(idx)))
                print('{} elapsed time: {:.3f}s'.format(time.ctime(),
                                                        time.time() - start_time))


def process():
    args = parse_args()
    if args.image_list != '':
        image_lists = read_file(args.image_lists)
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
    predict_image(args.root, output_root, args.checkpoint, category_index, image_lists, score, percent)
    compute_accuracy(os.path.join(args.root, 'gt_pre{}-{}.txt'.format(score, percent)))


if __name__ == '__main__':
    process()
