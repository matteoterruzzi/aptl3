import os
from urllib.request import pathname2url
import numpy as np

from ..images import getSmallImage, imageToArray


def test_get_small_image_to_array():
    _url = 'file:' + pathname2url(os.path.join(os.path.dirname(__file__), 'rose.jpg'))
    for _size in [1, 4, 224]:
        img = getSmallImage(_url, _size, _size)
        arr = imageToArray(img)
        assert tuple(arr.shape) == (_size, _size, 3)
        assert arr.dtype == np.uint8
        assert 0 <= np.min(arr) <= 200
        assert 129 < np.max(arr) <= 255
