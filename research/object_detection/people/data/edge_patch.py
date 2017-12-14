import os

from beer.data.tools import parse_args
from beer.data.tools import get_file_name
from beer.data.create_lists import create_train_val_list


def create_edge_patch_list(data_root, output_root, prefix='', postfix='', split='&!&'):
    file_list = os.listdir(data_root)
    output_list = []
    for file in file_list:
        path = os.path.join(data_root, file)
        if path[-2:] in SETS:
            annotations = os.listdir(os.path.join(path, 'annotation'))
            for ann in annotations:
                ann_path = os.path.join(path, 'annotation', ann)
                ex_img_path = os.path.join(path, 'image', '{}.jpg'.format(get_file_name(ann)))
                if os.path.exists(ex_img_path):
                    output_list.append(ex_img_path + split + ann_path)
    create_train_val_list(output_list, output_root, prefix, postfix)


if __name__ == '__main__':
    SETS = ['12', '13', '21', '24', '31', '34', '42', '43']
    args = parse_args()
    create_edge_patch_list(os.path.join(args.root_path, args.dataset),
                           args.root_path, postfix=args.postfix)
