import os

from beer.data.create_lists import create_train_val_list
from beer.data.create_lists import create_file_list
from beer.data.tools import ImageListCropper
from beer.data.tools import parse_args
from beer.utils.file_io import read_file
from beer.utils.file_io import read_label_as_list


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
