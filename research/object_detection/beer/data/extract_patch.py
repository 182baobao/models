import os
import cv2
import numpy as np

from beer.data.main import parse_args
from beer.data.create_lists import create_file_list
from beer.utils.file_io import read_label_as_list
from beer.utils.file_io import read_voc_xml
from beer.utils.file_io import write_file


def extract_patch_from_image(data_root, output_root, split='&!&'):
    img_list = []
    file_list, _ = create_file_list(data_root,
                                    params=(class_names, split))
    for idx, paths in enumerate(file_list):
        print('predicting {} of {} images'.format(idx, len(file_list)))
        print(paths)
        output_path = os.path.join(output_root, '{:04}'.format(idx // 1000),
                                   '{:08}'.format(idx))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img_path, xml_path = paths.split(split)
        xml_path = xml_path if xml_path[-1] == 'l' else xml_path[:-1]
        img = cv2.imread(img_path)
        objects, break_instance = read_voc_xml(xml_path, img.shape, class_names)
        if break_instance:
            continue
        for index, ob in enumerate(objects):
            sub_img = img[ob[2]:(ob[4] + 1), ob[1]:(ob[3] + 1), :]
            img_out_path = os.path.join(output_path,
                                        '{}_{}.jpg'.format(index, class_names.index(ob[0])))
            cv2.imwrite(img_out_path, sub_img)
            img_list.append(img_out_path + split + str(class_names.index(ob[0])))
    return img_list


def create_file_list_by_category(data_root, output_root,
                                 split='&!&', train_data_ratio=0.95):
    train_img_list = []
    val_img_list = []
    for idx, label in enumerate(class_names):
        img_root = os.path.join(data_root, label)
        img_list = os.listdir(img_root)
        category_list = []
        for img in img_list:
            img_path = os.path.join(img_root, img)
            category_list.append((img_path + split + str(idx)))
        np.random.shuffle(category_list)
        train = category_list[:int(train_data_ratio * len(category_list))]
        val = category_list[int(train_data_ratio * len(category_list)):]
        train_img_list += train
        val_img_list += val
    train_path = os.path.join(output_root, 'train{}.txt'.format(args.postfix))
    write_file(train_path, train_img_list)
    val_path = os.path.join(output_root, 'val{}.txt'.format(args.postfix))
    write_file(val_path, val_img_list)


if __name__ == '__main__':
    args = parse_args()
    class_names = read_label_as_list(args.label_file, args.class_num, args.instance)
    # create_train_val_list(extract_patch_from_image(os.path.join(args.root_path, args.dataset),
    #                                                os.path.join(args.root_path, args.target)),
    #                       args.root_path, args.postfix)
    create_file_list_by_category(os.path.join(args.root_path, args.dataset),
                                 args.root_path),
