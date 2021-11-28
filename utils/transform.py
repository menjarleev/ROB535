import random

import numpy as np
from skimage.transform import rotate
from scipy.ndimage import shift


def crop2d(img, psize):
    def _crop(_img, _psize, _ix, _iy):
        _img = _img[_iy:_iy + _psize, _ix:_ix + _psize, :]
        return _img
    h, w = img[0].shape[:-1]
    ix = random.randrange(0, w-psize+1)
    iy = random.randrange(0, h-psize+1)
    if type(img) == list:
        return [_crop(i, psize, ix, iy) for i in img]
    else:
        return _crop(img, psize, ix, iy)

def pixel_shift2d(img, shift_range=(-20, 20)):
    def _pixel_shift(_img, _vs, _hs):
        shift(_img, (_vs, _hs, 0))
        return np.clip(_img, 0, 255).astype(np.uint8)
    vshift = random.randrange(shift_range[0], shift_range[1])
    hshift = random.randrange(shift_range[0], shift_range[1])
    if type(img) == list:
        return [_pixel_shift(i, vshift, hshift) for i in img]
    else:
        return _pixel_shift(img, vshift, hshift)


def flip2d(img):
    def _flip(_img, _hflip, _vflip):
        if _hflip:
            _img = _img[:, ::-1, :]
        if _vflip:
            _img = _img[::-1, :, :]
        return _img
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    if type(img) == list:
        return [_flip(i, hflip, vflip) for i in img]
    else:
        return _flip(img, hflip, vflip)

def rotate2d(img):
    def _rot(_img, _degree):
        _img = rotate(_img, _degree, preserve_range=True)
        return _img
    degree = random.randint(0, 45)
    if type(img) == list:
        return [_rot(i, degree) for i in img]
    else:
        return _rot(img, degree)

def cutout(img, sz):
    def _cutout(_img, x, y, _sz):
        _img[y:y + _sz, x:x + _sz] = 0
        return _img
    (h, w, _) = img.shape
    x = random.randint(0, w - sz)
    y = random.randint(0, h - sz)
    if type(img) == list:
        return [_cutout(i, x, y, sz) for i in img]
    else:
        return _cutout(img, x, y, sz)

def shift_color(img):
    def _shift_color(_img, pmt):
        copy_img = _img.copy()
        copy_img[:, :, 0] = _img[:, :, pmt[0]]
        copy_img[:, :, 1] = _img[:, :, pmt[1]]
        copy_img[:, :, 2] = _img[:, :, pmt[2]]
        return copy_img
    pmt = np.random.permutation(3)
    if type(img) == list:
        return [_shift_color(i, pmt) for i in img]
    else:
        return _shift_color(img, pmt)

def guassian_noise(img, std=0.01):
    def _guassian_noise(_img, _std):
        noise = np.random.normal(0, _std, _img.shape)
        _img += noise
        _img[_img < 0] = 0
        _img[_img > 1] = 1
        return _img
    if type(img) == list:
        return [_guassian_noise(i, std) for i in img]
    else:
        return _guassian_noise(img, std)

class Transform:
    def __init__(self, transforms: list = []):
        self.transform = transforms

    def add_transform(self, transform):
        self.transform += [transform]

    def __call__(self, img):
        for trans in self.transform:
            img = trans(img)
        return img

def rotate3d(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)


def get_bbox(p0, p1):
    """
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    """
    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)

    return v, e
