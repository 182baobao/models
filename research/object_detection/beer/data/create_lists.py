import os
import numpy as np
import xml.etree.ElementTree as ET

from beer.utils.file_io import write_file


def get_file_name(filename):
    (_, temp_filename) = os.path.split(filename)
    (shot_name, _) = os.path.splitext(temp_filename)
    return shot_name


def _merge_dict(dict1, dict2):
    for key, value in dict2.items():
        if key not in dict1:
            dict1[key] = value
        else:
            dict1[key] += value
    return dict1


def _check_xml_file(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        sets = {}
        for obj in root.iter('object'):
            string = obj.find('name').text
            if string not in [
                    'Budweiser 600ML Bottle', 'Harbin Wheat 330ML Can',
                    'budweiser15', 'Budweiser Beer 500ML Can', 'harbin26',
                    'budweiser26', 'Budweiser Beer 330ML Can', 'budweiser31',
                    'budweiser30'
            ]:
                continue
            if string not in sets:
                sets[string] = 1
            else:
                sets[string] += 1
        return width > 0 and height > 0, sets
    except ET.ParseError:
        return False, {}


def _traverse_file(path, lists, sets):
    """
    """
    path = os.path.expanduser(path)
    list_files = os.listdir(path)
    for f in list_files:
        file_path = os.path.join(path, f)
        if os.path.isdir(file_path):
            lists, sets = _traverse_file(file_path, lists, sets)
        elif f.endswith('xml'):
            good, sets_ = _check_xml_file(file_path)
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


def create_train_val_list(data_root, output_root, postfix=''):
    lists = []
    sets = {}
    lists, sets = _traverse_file(data_root, lists, sets)
    np.random.shuffle(lists)
    train = lists[:int(0.95 * len(lists))]
    val = lists[int(0.95 * len(lists)):]
    train_path = os.path.join(output_root, 'train{}_list.txt'.format(postfix))
    write_file(train_path, train)
    val_path = os.path.join(output_root, 'val{}_list.txt'.format(postfix))
    write_file(val_path, val)
    return lists, sets


def create_file_list(data_root, output_file=''):
    lists = []
    sets = {}
    lists, sets = _traverse_file(data_root, lists, sets)
    if output_file != '':
        write_file(output_file, lists)
    return lists, sets


def count_instance_number(root, output_file):
    lists, sets = create_file_list(root)
    l_name = []
    l_num = []
    for key, value in sets.items():
        print('{:30}\t {}'.format(key, value))
        l_name.append(key)
        l_num.append(value)
    l_name = np.array(l_name)
    l_num = np.array(l_num)
    idx = np.argsort(l_num)
    l_num = l_num[idx]
    l_name = l_name[idx]
    output_set = np.hstack((l_num, l_name))
    np.save(output_file, output_set)


if __name__ == '__main__':
    out_root = '/home/admins/data/beer_data/data'
    count_instance_number(out_root, os.path.join(out_root, 'label.npy'))
