import numpy as np
import xml.etree.ElementTree as ET

from utils import label_map_util


def get_label_from_pd_file(pd_file, class_num):
    label_map = label_map_util.load_labelmap(pd_file)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=class_num, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def read_voc_xml(file_path, image_size):
    tree = ET.parse(file_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    break_instance = True
    objects = []

    def _read_xml(changed=False):
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            if cls_name not in [
                'Budweiser 600ML Bottle', 'Harbin Wheat 330ML Can',
                'budweiser15', 'Budweiser Beer 500ML Can', 'harbin26',
                'budweiser26', 'Budweiser Beer 330ML Can', 'budweiser31',
                'budweiser30'
            ]:
                continue
            xml_box = obj.find('bndbox')
            if not changed:
                xmin = int(xml_box.find('xmin').text)
                ymin = int(xml_box.find('ymin').text)
                xmax = int(xml_box.find('xmax').text)
                ymax = int(xml_box.find('ymax').text)
            else:
                ymin = int(xml_box.find('xmin').text)
                xmin = int(xml_box.find('ymin').text)
                ymax = int(xml_box.find('xmax').text)
                xmax = int(xml_box.find('ymax').text)
            if (0 <= xmin < xmax <= image_size[1]) or (
                    0 <= ymin < ymax <= image_size[0]):
                objects.append([cls_name, xmin, ymin, xmax, ymax])

    if (image_size[0] == height) and (image_size[1] == width):
        _read_xml()
        break_instance = False
    elif (image_size[0] == width) and (image_size[1] == height):
        _read_xml(True)
        break_instance = False
    return objects, break_instance


def read_file(root):
    info = []
    file = open(root, 'rt')
    while True:
        string = file.readline()
        if not string:
            break
        info.append(string)
    return info


def write_file(file_path, file_list1, file_list2=None):
    if file_list2 is None:
        with open(file_path, 'w') as file:
            for string in file_list1:
                print(string, file=file)
    else:
        if len(file_list1) == len(file_list2):
            with open(file_path, 'w') as file:
                for string1, string2 in zip(file_list1, file_list2):
                    print(string1 + ' ' + string2, file=file)
        else:
            return


def read_label_as_list(file_path, classes=9, instance=0):
    array = np.load(file_path)
    assert classes <= array.shape[0], 'required classes is bigger than actual !'
    assert instance > int(array[0, 0]), 'required instance is bigger than actual !'
    if instance == 0:
        return list(array[:classes, :])
    else:
        for idx, arr in enumerate(array):
            if int(arr[0]) < instance:
                return list(array[:(idx + 1), :])


def read_label_as_map_dict(file_path, classes=9, instance=0):
    label_list = read_label_as_list(file_path, classes, instance)
    map_dict = {}
    for idx, label in label_list:
        map_dict[str(idx + 1)] = {'id': idx + 1, 'name': label}
    return map_dict
