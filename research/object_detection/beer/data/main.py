import os
import argparse

from beer.data.create_lists import create_train_val_list
from beer.data.create_lists import create_file_list
from beer.data.tools import ImageListCropper
from beer.utils.file_io import read_file
from beer.utils.file_io import read_label_as_list


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare lists txt file for dataset')
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='dataset to use',
        default='data',
        type=str)
    parser.add_argument(
        '--target',
        dest='target',
        help='output list file',
        default='data',
        type=str)
    parser.add_argument(
        '--root',
        dest='root_path',
        help='dataset root path',
        default=os.path.join(os.getcwd(), 'data', 'beer'),
        type=str)
    parser.add_argument(
        '--postfix',
        dest='postfix',
        help='postfix to file',
        default='',
        type=str)
    parser.add_argument(
        '--label_file',
        dest='label_file',
        help='label file',
        default='',
        type=str)
    parser.add_argument(
        '--class_num',
        dest='class_num',
        help='class number',
        default=9,
        type=int)
    parser.add_argument(
        '--instance',
        dest='instance',
        help='required instance',
        default=0,
        type=int)
    return parser.parse_args()


def process_all(lists, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    for count, paths in enumerate(lists):
        print(paths)
        img_path, xml_path = paths.split('&!&')
        out_root = os.path.join(output_root, '{:04}'.format(count // 1000),
                                '{:08}'.format(count))
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        cropper = ImageListCropper(img_path, xml_path, out_root,
                                   cropped_size=(320, 320), stride=(80, 80))
        cropper.update(output_root + '/break.txt')


def _make_data():
    train = 'train{}'.format(args.postfix)
    val = 'val{}'.format(args.postfix)
    origin_data = os.path.join(args.root_path, args.dataset)
    output_data = os.path.join(args.root_path, args.target)
    file_list, _ = create_file_list(origin_data, label_list=label_list)
    create_train_val_list(file_list, args.root_path, args.postfix)
    train_list = read_file(os.path.join(args.root_path, '{}_list.txt'.format(train)))
    train_path = os.path.join(output_data, train)
    process_all(train_list, train_path)
    create_file_list(train_path,
                     os.path.join(args.root_path, '{}.txt'.format(train)), label_list)
    val_list = read_file(os.path.join(args.root_path, '{}_list.txt'.format(val)))
    val_path = os.path.join(output_data, val)
    process_all(val_list, val_path)
    create_file_list(val_path, os.path.join(args.root_path, '{}.txt'.format(val)), label_list)


if __name__ == '__main__':
    args = parse_args()
    label_list = read_label_as_list(args.label_file, args.class_num, args.instance)
    _make_data()
