import cv2
import numpy as np
from sklearn.metrics import average_precision_score
import xml.etree.ElementTree as ET


def read_img_xml_as_eval_info(img_path, xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    class_names = [
        'Harbin Wheat 330ML Can', 'Budweiser Beer 330ML Can',
        'Budweiser 600ML Bottle', 'Budweiser Beer 500ML Can', 'budweiser15',
        'budweiser31', 'budweiser30', 'harbin26',
        'budweiser26'
    ]
    info = {}
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    info['shape'] = (height, width)
    objects = []
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        if cls_name not in class_names:
            continue
        xml_box = obj.find('bndbox')
        xmin = int(xml_box.find('xmin').text) / width
        ymin = int(xml_box.find('ymin').text) / height
        xmax = int(xml_box.find('xmax').text) / width
        ymax = int(xml_box.find('ymax').text) / height
        objects.append([class_names.index(cls_name), xmin, ymin, xmax, ymax])
    info['objects'] = objects
    info['image'] = cv2.imread(img_path)
    return info


def is_overlap(rect1, rect2):
    return not ((rect1[0] > rect2[2]) or
                (rect1[1] > rect2[3]) or
                (rect1[2] < rect2[0]) or
                (rect1[3] < rect2[1]))


def get_overlap_area(rect1, rect2):
    xmin = max(rect1[0], rect2[0])
    ymin = max(rect1[1], rect2[1])
    xmax = min(rect1[2], rect2[2])
    ymax = min(rect1[3], rect2[3])
    return (xmax - xmin) * (ymax - ymin)


def compute_mean_average_precision(predictions, ground_true, top_k=0):
    end = predictions.size
    if top_k > 0:
        assert top_k < predictions.size, 'top_k is larger than predictions !'
        end = top_k
    aps = []
    for __i in range(1, end + 1):
        _aps = average_precision_score(ground_true[:__i], predictions[:__i])
        aps.append(_aps)
    return np.mean(aps)


if __name__ == '__main__':
    pre = np.random.rand(1000) * 0.75
    gt = np.array([1] * 620 + [0] * 380)
    pre = pre[np.argsort(pre)]
    print(pre)
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.9])
    print(average_precision_score(y_true, y_scores))
    print(compute_mean_average_precision(pre, gt))
