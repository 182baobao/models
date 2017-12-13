import tensorflow as tf


def read_and_decode(filename):
    """
    为了高效地读取数据，TF中使用队列（queue）读取数据。
    :param filename:
    :return:
    """
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.

    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.

    Args:
      image: an image of shape [height, width, channels].
      offset_height: a scalar tensor indicating the height offset.
      offset_width: a scalar tensor indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.

    Returns:
      the cropped (and resized) image.

    Raises:
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the data size.
    """
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the data size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    return tf.reshape(image, cropped_shape)


def random_crop(image_list, crop_height, crop_width):
    """Crops the given list of images.

    The function applies the same data to each image in the list. This can be
    effectively applied when there are multiple image inputs of the same
    dimension such as:

      image, depths, normals = _random_crop([image, depths, normals], 120, 150)

    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the new height.
      crop_width: the new width.

    Returns:
      the image_list with cropped images.

    Raises:
      ValueError: if there are multiple image inputs provided with different size
        or the images are smaller than the data dimensions.
    """
    if not image_list:
        raise ValueError('Empty image_list.')

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
    with tf.control_dependencies(asserts):
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width,
                  crop_height, crop_width) for image in image_list]


def central_crop(image_list, crop_height, crop_width):
    """Performs central crops of the given image list.

    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the height of the image following the data.
      crop_width: the width of the image following the data.

    Returns:
      the list of cropped images.
    """
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs


def mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel.

    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.

    Returns:
      the centered image.

    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def _smallest_size_at_least(height, width, smallest_side):
    """Computes new shape with the smallest side equal to `smallest_side`.

    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.

    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: and int32 scalar tensor indicating the new width.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    return new_height, new_width


def aspect_preserving_resize(image, smallest_side):
    """Resize images preserving the original aspect ratio.

    Args:
      image: A 3-D image `Tensor`.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      resized_image: A 3-D tensor containing the resized image.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                             align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image


def central_padding(image, output_shape):
    """
    make central padding to the image,
    :param image: A `Tensor` representing an image of arbitrary size, the
                shape should be (H, W, C)
    :param output_shape: the output image tensor shape,
                all of the dimension in output_shape must not smaller than in
                image's shape
    :return: A `Tensor` with the shape of output_shape
    """
    shape = tf.shape(tf.squeeze(image))
    pad_head = tf.subtract(output_shape, shape) // 2
    pad_tail = tf.subtract(output_shape, shape) - pad_head
    pad_head = tf.reshape(pad_head, [3, 1])
    pad_tail = tf.reshape(pad_tail, [3, 1])
    padding = tf.concat([pad_head, pad_tail], 1)
    return tf.pad(image, padding)


def resize_larger_dimension(image, output_height, output_width):
    """
    if one or some dimensions of image are bigger than output shape,
    resize the image so that the biggest dimension to be the same as
    the corresponding dimension
    :param image: A `Tensor` representing an image of arbitrary size, the
                shape should be (H, W, C)
    :param output_height: The biggest height of the image after preprocessing.
    :param output_width: The biggest width of the image after preprocessing.
    :return:
    """
    height = tf.constant(output_height)
    width = tf.constant(output_width)
    flag = [0]

    def _app(f, dim=False, value=0):
        if dim:
            f.append(value)
        return True

    while True:
        shape = tf.shape(image)
        tf.cond((tf.squeeze(shape[0]) <= height) & (tf.squeeze(shape[1]) <= width),
                lambda: _app(flag, True), lambda: _app(flag))
        if flag[-1] == 0:
            break
        min_dim = tf.to_float(tf.minimum(shape[0], shape[1]))
        tf.cond(tf.squeeze(shape[0]) > height,
                lambda: _app(flag), lambda: _app(flag, True, 1))
        if flag[-1] != 1:
            smallest_side = min_dim * height / shape[0]
            image = aspect_preserving_resize(image, tf.to_int32(smallest_side))
        tf.cond(tf.squeeze(shape[1]) > height,
                lambda: _app(flag), lambda: _app(flag, True, 2))
        if flag[-1] != 2:
            smallest_side = min_dim * width / shape[1]
            image = aspect_preserving_resize(image, tf.to_int32(smallest_side))
    return image


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min,
                         resize_side_max):
    """Preprocesses the given image for training.

    Note that the actual resizing scale is sampled from
      [`resize_size_min`, `resize_size_max`].

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      resize_side_min: The lower bound for the smallest side of the image for
        aspect-preserving resizing.
      resize_side_max: The upper bound for the smallest side of the image for
        aspect-preserving resizing.

    Returns:
      A preprocessed image.
    """
    resize_side = tf.random_uniform(
        [], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)
    image = aspect_preserving_resize(image, resize_side)
    shape = tf.shape(image)
    image = random_crop([image], shape[0], shape[1])[0]
    image = tf.reshape(image, [shape[0], shape[1], 3])
    image = resize_larger_dimension(image, output_height, output_width)
    image = central_padding(image, (output_height, output_width, 3))
    image = tf.to_float(image)
    # return mean_image_subtraction(image, [0, 0, 0])
    return tf.reshape(image, [output_height, output_width, 3])


def preprocess_for_eval(image, output_height, output_width, resize_side):
    """Preprocesses the given image for evaluation.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      resize_side: The smallest side of the image for aspect-preserving resizing.

    Returns:
      A preprocessed image.
    """
    image = aspect_preserving_resize(image, resize_side)
    shape = tf.shape(image)
    image = central_crop([image], shape[0], shape[1])[0]
    image = tf.reshape(image, [shape[0], shape[1], 3])
    image = resize_larger_dimension(image, output_height, output_width)
    image = central_padding(image, (output_height, output_width, 3))
    image = tf.to_float(image)
    # return mean_image_subtraction(image, [0, 0, 0])
    return tf.reshape(image, [output_height, output_width, 3])


def preprocess_image(image, output_height, output_width, is_training=False, scale=0.1):
    shape = tf.shape(image)
    min_dim = tf.to_float(tf.minimum(shape[0], shape[1]))
    resize_side_min = tf.to_int32(tf.round(min_dim * tf.convert_to_tensor([1.0 - scale])))
    resize_side_max = tf.to_int32(tf.round(min_dim * tf.convert_to_tensor([1.0 + scale])))
    if is_training:
        return preprocess_for_train(image, output_height, output_width,
                                    resize_side_min, resize_side_max)
    else:
        return preprocess_for_eval(image, output_height, output_width,
                                   resize_side_min)
