# Module implements smoothing using cross-correlation.
import cv2
import numpy as np
from skimage.registration import phase_cross_correlation


def extract(rect_orig):
    rect = cv2.cvtColor(rect_orig, cv2.COLOR_RGB2GRAY)

    shape_rect = max(rect_orig.shape[0], rect_orig.shape[1])
    ksize = shape_rect // 100
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    rect = cv2.blur(rect, (ksize, ksize))
    return rect


def get_shift(prev_rect, current_rect_with_noise):
    rect_curr = extract(current_rect_with_noise)
    rect_prev = extract(prev_rect)

    rect_curr_f = cv2.flip(cv2.flip(rect_curr, 1), 0)
    rect_prev_f = cv2.flip(cv2.flip(rect_prev, 1), 0)

    mx_sh = np.max([rect_prev.shape, rect_curr.shape], 0)
    mx_sh_f = np.max([rect_prev_f.shape, rect_curr_f.shape], 0)

    # padding to one size
    rect_prev = np.pad(rect_prev, [(0, mx_sh[0] - rect_prev.shape[0]), (0, mx_sh[1] - rect_prev.shape[1])],
                       'constant')
    rect_curr = np.pad(rect_curr, [(0, mx_sh[0] - rect_curr.shape[0]), (0, mx_sh[1] - rect_curr.shape[1])],
                       'constant')
    rect_prev_f = np.pad(rect_prev_f,
                         [(0, mx_sh_f[0] - rect_prev_f.shape[0]), (0, mx_sh_f[1] - rect_prev_f.shape[1])],
                         'constant')
    rect_curr_f = np.pad(rect_curr_f,
                         [(0, mx_sh_f[0] - rect_curr_f.shape[0]), (0, mx_sh_f[1] - rect_curr_f.shape[1])],
                         'constant')
    shift, err0, phdiff = phase_cross_correlation(rect_prev, rect_curr,
                                                  normalization="phase")
    shift_f, err1, phdiff = phase_cross_correlation(rect_prev_f, rect_curr_f,
                                                    normalization="phase")
    return shift, shift_f


def extract_patch(image, bbox):
    x, y, w, h = bbox
    return image[y:y + h, x:x + w]


def smooth_bbox(prev_rect, current_rect_with_noise, pred_bbox):
    x0, y0, w, h = pred_bbox
    x1, y1 = x0 + w, y0 + h

    shift, shift_f = get_shift(prev_rect, current_rect_with_noise)
    shift = np.clip(shift, -20, 20)
    shift_f = np.clip(shift_f, -20, 20)
    # print(shift, shift_f)
    x0, x1, y0, y1 = int(x0 - shift[1]), int(x1 + shift_f[1]), int(y0 - shift[0]), int(y1 + shift_f[0])
    return [x0, y0, x1 - x0, y1 - y0], shift, shift_f
