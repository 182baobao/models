import os

from beer.data.create_lists import create_train_val_list
from beer.data.create_lists import create_file_list
from beer.data.tools import ImageListCropper
from beer.data.tools import parse_args
from beer.utils.file_io import read_file
from beer.utils.file_io import read_label_as_list


def process_all(lists, output_root, split='&!&'):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    for count, paths in enumerate(lists):
        print(paths)
        img_path, xml_path = paths.split(split)
        out_root = os.path.join(output_root, '{:04}'.format(count // 1000),
                                '{:08}'.format(count))
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        cropper = ImageListCropper(img_path, xml_path, out_root,
                                   cropped_size=(320, 320), stride=(80, 80))
        cropper.update(output_root + '/break.txt')


def _make_data(prefix, postfix, split='&!&'):
    train = 'train{}'.format(postfix)
    val = 'val{}'.format(postfix)
    origin_data = os.path.join(args.root_path, args.dataset)
    output_data = os.path.join(args.root_path, args.target)
    file_list, _ = create_file_list(origin_data, params=(class_names, split))
    create_train_val_list(file_list, args.root_path, prefix, args.postfix)
    train_list = read_file(os.path.join(args.root_path, '{}{}.txt'.format(prefix, train)))
    train_path = os.path.join(output_data, train)
    process_all(train_list, train_path, split)
    create_file_list(train_path,
                     os.path.join(args.root_path, '{}.txt'.format(train)),
                     params=(class_names, split))
    val_list = read_file(os.path.join(args.root_path, '{}{}.txt'.format(prefix, val)))
    val_path = os.path.join(output_data, val)
    process_all(val_list, val_path, split)
    create_file_list(val_path, os.path.join(args.root_path, '{}.txt'.format(val)), class_names)


if __name__ == '__main__':
    args = parse_args()
    class_names = read_label_as_list(args.label_file, args.class_num, args.instance)
    _make_data(args.prefix, args.postfix)
