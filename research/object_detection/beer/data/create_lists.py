import os
import numpy as np
import xml.etree.ElementTree as ET

from beer.data.tools import parse_args
from beer.data.tools import get_file_name
from beer.utils.file_io import write_file


def _merge_dict(dict1, dict2):
    for key, value in dict2.items():
        if key not in dict1:
            dict1[key] = value
        else:
            dict1[key] += value
    return dict1


def _check_xml_file(file_path, label_list):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        sets = {}
        for obj in root.iter('object'):
            string = obj.find('name').text
            if len(label_list) > 0:
                if string not in label_list:
                    continue
            if string not in sets:
                sets[string] = 1
            else:
                sets[string] += 1
        return width > 0 and height > 0, sets
    except ET.ParseError:
        return False, {}


def _traverse_file(path, lists, sets, label_list):
    """
    """
    path = os.path.expanduser(path)
    list_files = os.listdir(path)
    for f in list_files:
        file_path = os.path.join(path, f)
        if os.path.isdir(file_path):
            lists, sets = _traverse_file(file_path, lists, sets, label_list)
        elif f.endswith('xml'):
            good, sets_ = _check_xml_file(file_path, label_list)
            print(file_path)
            if (not good) or (len(sets_) == 0):
                continue
            sets = _merge_dict(sets, sets_)
            image_path = os.path.join(path, get_file_name(f))
            if os.path.exists(image_path + '.jpg'):
                img_path = image_path + '.jpg'
            elif os.path.exists(image_path + '.png'):
                img_path = image_path + '.png'
            elif os.path.exists(image_path + '.JPG'):
                img_path = image_path + '.JPG'
            elif os.path.exists(image_path + '.PNG'):
                img_path = image_path + '.PNG'
            else:
                continue
            lists.append(img_path + '&!&' + file_path)
        else:
            continue
    return lists, sets


def create_train_val_list(file_list, output_root, prefix='',
                          postfix='', train_data_ratio=0.95):
    np.random.shuffle(file_list)
    train = file_list[:int(train_data_ratio * len(file_list))]
    val = file_list[int(train_data_ratio * len(file_list)):]
    train_path = os.path.join(output_root, '{}train{}.txt'.format(prefix, postfix))
    write_file(train_path, train)
    val_path = os.path.join(output_root, '{}val{}.txt'.format(prefix, postfix))
    write_file(val_path, val)


def create_file_list(data_root,
                     output_file='', label_list=[]):
    lists = []
    sets = {}
    lists, sets = _traverse_file(data_root, lists, sets, label_list)
    if output_file != '':
        write_file(output_file, lists)
    return lists, sets


def count_instance_number(root, output_file, split='&!&'):
    lists, sets = create_file_list(root)
    l_name = []
    l_num = []
    for key, value in sets.items():
        print('{:30}\t {}'.format(key, value))
        if key == 'Others':
            continue
        l_name.append(key)
        l_num.append(value)
    l_name = np.array(l_name)
    l_num = np.array(l_num)
    idx = np.argsort(-l_num)
    l_num = list(map(lambda x: str(x), l_num[idx]))
    l_name = l_name[idx]
    write_file(output_file, list(l_num), list(l_name), split)


if __name__ == '__main__':
    args = parse_args()
    count_instance_number(os.path.join(args.root_path, args.dataset),
                          os.path.join(args.target, 'beer_label.txt'))
