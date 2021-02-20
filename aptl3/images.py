import base64
import io
import os
import subprocess
import warnings
from urllib.request import urlopen, url2pathname

from PIL import Image, ImageOps, UnidentifiedImageError
from numpy import asarray as np_asarray

ImageError = UnidentifiedImageError


def getImage(fp):
    """
    :param fp: Accepts local file paths, direct bytes data or file object
    """
    if isinstance(fp, str):
        if (
                fp.startswith('http:') or
                fp.startswith('https:') or
                fp.startswith('ftp:') or
                fp.startswith('data:')):
            warnings.warn("Opening remote URL")
            fp = urlopen(fp)
        else:
            # assert fp is local file
            if fp.startswith('file:'):
                fp = url2pathname(fp.split('file:', 1)[1])
            if isRawImageFile(fp):
                fp = io.BytesIO(extractPreviewFromRaw(fp))
    elif isinstance(fp, bytes):
        fp = io.BytesIO(fp)
    return ImageOps.exif_transpose(Image.open(fp, mode='r'))


def getSmallImage(fp, width=300, height=300):
    im = getImage(fp)
    im = ImageOps.fit(im, (width, height))
    return im


def imageToDataURL(image):
    with io.BytesIO() as output:
        fmt = 'PNG' if image.info.get('transparency') is not None or image.mode == 'RGBA' else 'JPEG'
        image.save(output, format=fmt)
        return "data:image/"+fmt.lower()+";base64," + base64.b64encode(output.getvalue()).decode()


def imageToBytesIO(image) -> io.BytesIO:
    output = io.BytesIO()
    fmt = 'PNG' if image.info.get('transparency') is not None or image.mode == 'RGBA' else 'JPEG'
    image.save(output, format=fmt)
    return output


def imageFromArray(im):
    """Requires a uint8 numpy array in the 0-255 scale. """
    return Image.fromarray(im)


def imageToArray(im):
    return np_asarray(im.convert('RGB'))  # converts from grayscale to RGB and also removed alpha if present


def isImageFile(path: str):
    filename, ext = os.path.splitext(path)
    ext = ext[1:].lower()
    return ext in ["jpg", "jpeg", "png", "gif", "nef", "cr2", "crw"]


def isRawImageFile(path: str):
    filename, ext = os.path.splitext(path)
    ext = ext[1:].lower()
    return ext in ["nef", "cr2", "crw"]


def verifyImageFile(path: str):
    _size = os.path.getsize(path)
    if _size < 1000:
        raise Exception('File too short: %d bytes.' % _size)
    if _size > 1000000:
        return
    getSmallImage(path, 4, 4)
    # im = Image.open(path, mode='r')
    # im.verify()


def extractPreviewFromRaw(filePath: str):
    # TODO: use https://pypi.org/project/pyunraw/ writing on temp file
    if filePath.startswith('file:'):
        filePath = filePath[len('file:'):]
    # WARNING: if you remove -c, then dcraw will write a file on your hard disk. (-e is to extract embedded preview)
    out = subprocess.check_output(['dcraw', '-c', '-e', filePath])
    if len(out) <= 0:
        raise Exception('Cannot decode RAW image')
    return out
