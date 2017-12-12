import os
import cv2

from beer.data.main import parse_args
from beer.data.create_lists import create_file_list
from beer.data.create_lists import create_train_val_list
from beer.utils.file_io import read_label_as_list
from beer.utils.file_io import read_voc_xml


def extract_patch():
    img_list = []
    file_list, _ = create_file_list(os.path.join(args.root_path, args.dataset),
                                    label_list=label_list)
    output_root = os.path.join(args.root_path, args.target)
    for idx, paths in enumerate(file_list):
        print('predicting {} of {} images'.format(idx, len(file_list)))
        print(paths)
        output_path = os.path.join(output_root, '{:04}'.format(idx // 1000),
                                   '{:08}'.format(idx))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img_path, xml_path = paths.split('&!&')
        xml_path = xml_path if xml_path[-1] == 'l' else xml_path[:-1]
        img = cv2.imread(img_path)
        objects, break_instance = read_voc_xml(xml_path, img.shape, label_list)
        if break_instance:
            continue
        for index, ob in enumerate(objects):
            sub_img = img[ob[2]:(ob[4] + 1), ob[1]:(ob[3] + 1), :]
            img_out_path = os.path.join(output_path,
                                        '{}_{}.jpg'.format(index, label_list.index(ob[0])))
            cv2.imwrite(img_out_path, sub_img)
            img_list.append(img_out_path + '&!&' + str(label_list.index(ob[0])))
    return img_list


if __name__ == '__main__':
    args = parse_args()
    label_list = read_label_as_list(args.label_file, args.class_num, args.instance)
    create_train_val_list(extract_patch(), args.root_path, args.postfix)
