import xml.etree.ElementTree as ET
import cv2
import os


def _add_element(_root, _name, _value):
    sub_element = ET.SubElement(_root, _name)
    sub_element.text = _value


def _read_xml(file_path, image_size):
    tree = ET.parse(file_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    break_instance = True
    objects = []

    def __read_xml(changed=False):
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            if cls_name not in [
                'Budweiser 600ML Bottle', 'Harbin Wheat 330ML Can',
                'budweiser15', 'Budweiser Beer 500ML Can', 'harbin26',
                'budweiser26', 'Budweiser Beer 330ML Can', 'budweiser31',
                'budweiser30'
            ]:
                continue
            xml_box = obj.find('bndbox')
            if not changed:
                xmin = int(xml_box.find('xmin').text)
                ymin = int(xml_box.find('ymin').text)
                xmax = int(xml_box.find('xmax').text)
                ymax = int(xml_box.find('ymax').text)
            else:
                ymin = int(xml_box.find('xmin').text)
                xmin = int(xml_box.find('ymin').text)
                ymax = int(xml_box.find('xmax').text)
                xmax = int(xml_box.find('ymax').text)
            if (0 <= xmin < xmax <= image_size[1]) or (
                    0 <= ymin < ymax <= image_size[0]):
                objects.append([cls_name, xmin, ymin, xmax, ymax])

    if (image_size[0] == height) and (image_size[1] == width):
        __read_xml()
        break_instance = False
    elif (image_size[0] == width) and (image_size[1] == height):
        __read_xml(True)
        break_instance = False
    return objects, break_instance


class SubImageCropper(object):
    """
    get sub image from big image
    """

    def __init__(self, image_path,
                 cropped_size=(416, 416),
                 stride=(104, 104)):
        self._image = cv2.imread(image_path)
        self._cropped_size = list(cropped_size)
        self._widths = []
        self._in_widths = []
        self._heights = []
        self._in_heights = []
        self._get_crop_image_seats(stride, self._image.shape)

    def _get_crop_image_seats(self, stride, image_size):
        self._cropped_size[1] = min(self._cropped_size[1], image_size[1])
        self._cropped_size[0] = min(self._cropped_size[0], image_size[0])
        self._widths = list(
            range(0, image_size[1] - self._cropped_size[1] + 1, stride[1]))
        self._heights = list(
            range(0, image_size[0] - self._cropped_size[0] + 1, stride[0]))
        self._in_widths = list(
            map(lambda x: x + image_size[1] % stride[1], self._widths))
        self._in_heights = list(
            map(lambda x: x + image_size[0] % stride[0], self._heights))

    def _get_sub_image(self, *args, **kwargs):
        raise NotImplementedError('this method is not implemented !')

    def _preprocess(self, *args, **kwargs):
        if len(args) > 0 or len(kwargs) > 0:
            raise ValueError('no need parameters !')
        return len(self._image.shape) < 2

    @property
    def cropped_size(self):
        return self._cropped_size

    @property
    def image(self):
        return self._image

    def update(self, *args, **kwargs):
        if self._preprocess(*args, **kwargs):
            return
        print('cropping...')
        for x in self._widths:
            for y in self._heights:
                self._get_sub_image(x, y)

        for x in self._in_widths:
            for y in self._in_heights:
                self._get_sub_image(x, y)


class ImageDictCropper(SubImageCropper):
    """
    crop image and get image array list
    """

    def __init__(self, image_path,
                 cropped_size=(416, 416),
                 stride=(104, 104)):
        super(ImageDictCropper, self).__init__(image_path,
                                               cropped_size,
                                               stride)
        self._image_dict = {}

    def _get_sub_image(self, x, y):
        xmax = x + self.cropped_size[0]
        ymax = y + self.cropped_size[1]
        sub_image = self.image[y:ymax, x:xmax, :]
        self._image_dict['{}_{}'.format(x, y)] = sub_image

    def get_images(self):
        return self._image_dict


class ImageListCropper(SubImageCropper):
    """
    crop_tool the beer image dataset
    """

    def __init__(self,
                 image_path,
                 xml_path,
                 output_root,
                 cropped_size=(416, 416),
                 stride=(104, 104),
                 threshold=0.8):
        super(ImageListCropper, self).__init__(image_path,
                                               cropped_size,
                                               stride)
        self._image_path = image_path
        self._xml_path = xml_path
        self._output_root = output_root
        self._threshold = threshold
        self._objects = []

    def _get_sub_image(self, x, y):
        h_list = list(range(y, y + self.cropped_size[1]))
        w_list = list(range(x, x + self.cropped_size[0]))
        output_objects = []
        for ob in self._objects:
            if (ob[1] in w_list) and (ob[3] in w_list) and (
                    ob[2] in h_list) and (ob[4] in h_list):
                output_objects.append(
                    [ob[0], ob[1] - x, ob[2] - y, ob[3] - x, ob[4] - y])
            elif (ob[1] < w_list[0]) or (ob[3] > w_list[-1]) or (
                    ob[2] < h_list[0]) or (ob[4] > h_list[-1]):
                continue
            else:
                xmin = ob[1] if (ob[1] in w_list) else w_list[0]
                ymin = ob[2] if (ob[2] in h_list) else h_list[0]
                xmax = ob[3] if (ob[3] in w_list) else w_list[-1]
                ymax = ob[4] if (ob[4] in h_list) else h_list[-1]
                area = (xmax - xmin) * (ymax - ymin)
                ob_area = (ob[3] - ob[1]) * (ob[4] - ob[2])
                if (area / ob_area) >= self._threshold:
                    output_objects.append(
                        [ob[0], xmin - x, ymin - y, xmax - x, ymax - y])
        if len(output_objects) > 0:
            sub_image = self.image[h_list[0]:h_list[-1], w_list[0]:w_list[
                -1], :]
            self._write_image_and_object(sub_image, output_objects,
                                         os.path.join(self._output_root,
                                                      '{}_{}'.format(
                                                          x, y)))

    def _write_image_and_object(self, image, objects, file_name):
        cv2.imwrite(file_name + '.jpg', image)
        root = ET.Element('annotation')
        _add_element(root, 'src_img', self._image_path)
        _add_element(root, 'xml_path', self._xml_path)
        size = ET.SubElement(root, 'size')
        _add_element(size, 'src_height', str(self.image.shape[0]))
        _add_element(size, 'src_width', str(self.image.shape[0]))
        _add_element(size, 'height', str(self.cropped_size[1]))
        _add_element(size, 'width', str(self.cropped_size[0]))
        _add_element(size, 'depth', '3')
        for ob in objects:
            ob_xml = ET.SubElement(root, 'object')
            _add_element(ob_xml, 'name', ob[0])
            _add_element(ob_xml, 'difficult', '0')
            bndbox = ET.SubElement(ob_xml, 'bndbox')
            _add_element(bndbox, 'xmin', str(ob[1]))
            _add_element(bndbox, 'ymin', str(ob[2]))
            _add_element(bndbox, 'xmax', str(ob[3]))
            _add_element(bndbox, 'ymax', str(ob[4]))
        tree = ET.ElementTree(root)
        tree.write(file_name + '.xml')

    def _preprocess(self, break_image=''):
        objects, break_instance = _read_xml(self._xml_path,
                                            self.image.shape)
        self._objects = objects[:]
        if break_instance or (len(self._objects) == 0):
            if break_image != '':
                with open(break_image, 'a') as out:
                    print(self._image_path, file=out)
            return True
        else:
            return False

# class ImageCropper(object):
#     """
#     crop_tool the beer image dataset
#     """
#
#     def __init__(self,
#                  image_path,
#                  xml_path,
#                  output_root,
#                  cropped_size=(416, 416),
#                  stride=(104, 104),
#                  threshold=0.8):
#         self.image = cv2.imread(image_path)
#         self.image_path = image_path
#         self.xml_path = xml_path
#         self.output_root = output_root
#         self.cropped_size = cropped_size
#         self.threshold = threshold
#         self.widths = []
#         self.in_widths = []
#         self.heights = []
#         self.in_heights = []
#         self._get_crop_image_seats(stride, self.image.shape)
#
#     def _get_crop_image_seats(self, stride, image_size):
#         self.cropped_size[1] = min(self.cropped_size[1], image_size[1])
#         self.cropped_size[0] = min(self.cropped_size[0], image_size[0])
#         self.widths = list(
#             range(0, image_size[1] - self.cropped_size[1] + 1, stride[1]))
#         self.heights = list(
#             range(0, image_size[0] - self.cropped_size[0] + 1, stride[0]))
#         self.in_widths = list(
#             map(lambda x: x + image_size[1] % stride[1], self.widths))
#         self.in_heights = list(
#             map(lambda x: x + image_size[0] % stride[0], self.heights))
#
#     def _get_sub_image(self, x, y, objects):
#         h_list = list(range(y, y + self.cropped_size[1] + 1))
#         w_list = list(range(x, x + self.cropped_size[0] + 1))
#         output_objects = []
#         for ob in objects:
#             if (ob[1] in w_list) and (ob[3] in w_list) and (
#                     ob[2] in h_list) and (ob[4] in h_list):
#                 output_objects.append(
#                     [ob[0], ob[1] - x, ob[2] - y, ob[3] - x, ob[4] - y])
#             elif (ob[1] < w_list[0]) or (ob[3] > w_list[-1]) or (
#                     ob[2] < h_list[0]) or (ob[4] > h_list[-1]):
#                 continue
#             else:
#                 xmin = ob[1] if (ob[1] in w_list) else w_list[0]
#                 ymin = ob[2] if (ob[2] in h_list) else h_list[0]
#                 xmax = ob[3] if (ob[3] in w_list) else w_list[-1]
#                 ymax = ob[4] if (ob[4] in h_list) else h_list[-1]
#                 area = (xmax - xmin) * (ymax - ymin)
#                 ob_area = (ob[3] - ob[1]) * (ob[4] - ob[2])
#                 if (area / ob_area) >= self.threshold:
#                     output_objects.append(
#                         [ob[0], xmin - x, ymin - y, xmax - x, ymax - y])
#         if len(output_objects) > 0:
#             sub_image = self.image[h_list[0]:h_list[-1], w_list[0]:w_list[
#                 -1], :]
#             self._write_image(sub_image, output_objects,
#                               os.path.join(self.output_root, '{}_{}'.format(
#                                   x, y)))
#
#     def _write_image(self, image, objects, file_name):
#         cv2.imwrite(file_name + '.jpg', image)
#         root = ET.Element('annotation')
#
#         src_img = ET.SubElement(root, 'src_img')
#         src_img.text = self.image_path
#         xml_path = ET.SubElement(root, 'xml_path')
#         xml_path.text = self.xml_path
#
#         size = ET.SubElement(root, 'size')
#
#         src_height = ET.SubElement(root, 'src_height')
#         src_height.text = str(self.cropped_size[1])
#         src_width = ET.SubElement(root, 'src_width')
#         src_width.text = str(self.cropped_size[0])
#
#         height = ET.SubElement(size, 'height')
#         height.text = str(self.cropped_size[1])
#         width = ET.SubElement(size, 'width')
#         width.text = str(self.cropped_size[0])
#             ob_xml = ET.SubElement(root, 'object')
#             name = ET.SubElement(ob_xml, 'name')
#             name.text = ob[0]
#             difficult = ET.SubElement(ob_xml, 'difficult')
#             difficult.text = '0'
#             bndbox = ET.SubElement(ob_xml, 'bndbox')
#             xmin = ET.SubElement(bndbox, 'xmin')
#             xmin.text = str(ob[1])
#             ymin = ET.SubElement(bndbox, 'ymin')
#             ymin.text = str(ob[2])
#             xmax = ET.SubElement(bndbox, 'xmax')
#             xmax.text = str(ob[3])
#             ymax = ET.SubElement(bndbox, 'ymax')
#             ymax.text = str(ob[4])
#         tree = ET.ElementTree(root)
#         tree.write(file_name + '.xml')
#
#     def update(self, break_image=''):
#         objects, break_instance = _read_xml(self.xml_path, self.image.shape)
#         if break_instance or (len(objects) == 0):
#             if break_image != '':
#                 with open(break_image, 'a') as out:
#                     print(self.image_path, file=out)
#             return
#         print('cropping...', len(objects))
#         for x in self.widths:
#             for y in self.heights:
#                 self._get_sub_image(x, y, objects)
#
#         for x in self.in_widths:
#             for y in self.in_heights:
#                 self._get_sub_image(x, y, objects)
