import os

from beer.eval_tool.tools import read_img_xml_as_eval_info
from beer.eval_tool.detector import BeerDetector
from beer.eval_tool.tools import compute_mean_average_precision
from beer.crop_tool.create_lists import create_file_list
from beer.crop_tool.tools import ImageListCropper

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare lists txt file for dataset')
    parser.add_argument(
        '--pd_file',
        dest='pd_file',
        default='',
        type=str)
    parser.add_argument(
        '--target',
        dest='target',
        help='output list file',
        default='pre',
        type=str)
    parser.add_argument(
        '--root',
        dest='root_path',
        help='dataset root path',
        default='',
        type=str)
    args = parser.parse_args()
    return args


def process_all(lists, output_root, pd_file):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    detector = BeerDetector(pd_file)
    objects = []
    curr_path = os.path.abspath(os.path.dirname(__file__))
    for count, paths in enumerate(lists):
        print(paths)
        img_path, xml_path = paths.split('&!&')
        info = read_img_xml_as_eval_info(img_path, xml_path)
        info['crop_shape'] = (416, 416)
        out_root = os.path.join(output_root, '{:04}'.format(count // 1000),
                                '{:08}'.format(count))
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        out_file_root = os.path.join(output_root, '{:04}'.format(count // 1000))
        cropper = ImageListCropper(img_path, xml_path, out_root)
        cropper.update(output_root + '/break.txt')
        images, _ = create_file_list(out_root, out_file_root + '/img.txt')
        img_list = list(map(lambda x: x.split('&!&')[0], images))
        detector.detect_images(img_list, info)
        objects += detector.get_result()
        detector.visualize(os.path.join(curr_path, '..', '..', 'data', 'beer_label_map.pbtxt',
                                        os.path.join(out_root, '{}.png'.format(count))))
    return objects


if __name__ == '__main__':
    args = parse_args()
    lists, _ = create_file_list(args.root_path, args.root_path + '/img.txt')
    objects = process_all(lists, args.target, args.pd_file)
    print(compute_mean_average_precision(objects))
