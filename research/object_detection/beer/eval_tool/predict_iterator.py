import os
from PIL import Image
import numpy as np

from beer.eval_tool.tools import read_img_xml_as_eval_info


class ImageIter(object):
    """
    a iterator of image list
    """

    def __init__(self, img_list):
        self._img_list = img_list
        self._location = 0

    @property
    def location(self):
        return self._location

    def has_next(self):
        return self._location < len(self._img_list)

    def next(self):
        next_img = self._img_list[self._location]
        self._location += 1
        return self._get_next(next_img)

    def _get_next(self, next_img):
        return [{'img': next_img}]

    def set_predict(self, *args, **kwargs):
        self._process_predict(*args, **kwargs)

    def _process_predict(self, *args, **kwargs):
        raise NotImplementedError('')


class SmallImageIter(ImageIter):
    """
    predict small image iterator
    """

    def __init__(self, img_list, score, threshold):
        super(SmallImageIter, self).__init__(img_list)
        self._score = score
        self._threshold = threshold

    def _get_next(self, next_img):
        img_path, xml_path = next_img.split('&!&')
        image = Image.open(img_path)
        image_np = np.array(image).astype(np.uint8)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        info = read_img_xml_as_eval_info(img_path, xml_path[:-1])
        return [{'img': image_np_expanded, 'info': info}]

    def _process_predict(self, info_dict):
        pic = Image.fromarray(info_dict['img_arr'])
        pic.save(os.path.join(info_dict['output_root'], '{}.jpg'.format(self.location)))
