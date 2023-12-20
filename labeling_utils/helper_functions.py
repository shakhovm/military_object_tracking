import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
from image_preprocessor import ImagePreprocessor


class Helper:
    def __init__(self, cap_path=None, mask_cap_path=None):
        self.cap = cv2.VideoCapture(cap_path)
        self.cap_mask = cv2.VideoCapture(mask_cap_path)
        self.im = ImagePreprocessor()

    @staticmethod
    def _get_img(freq_seq, cap):
        cap.set(1, freq_seq)
        ret, img = cap.read()
        return img

    def get_img(self, freq_seq):
        return cv2.cvtColor(self._get_img(freq_seq, self.cap), cv2.COLOR_BGR2RGB)

    def get_bgr(self, freq_seq):
        return self._get_img(freq_seq, self.cap)

    def get_mask(self, freq_seq):
        return self._get_img(freq_seq, self.cap_mask)

    def resize(self, img, scale):
        return cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
    #

    def to_rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def bgr_show(self, img):
        plt.imshow(self.to_rgb(img))
        plt.show()

    def imshow(self, img):
        plt.imshow(img)
        plt.show()

    def display_bboxes(self, img, boxes):
        colors = [(0,0,255), (255, 0, 0)]
        if len(boxes) > 0:
            if isinstance(boxes[0], float):
                rect_bbox = list(map(round, boxes[:4]))
                cv2.rectangle(img, (rect_bbox[0], rect_bbox[1]), (rect_bbox[2], rect_bbox[3]), color=(0, 0, 255),
                              thickness=2)
                return
            for i, bbox in enumerate(boxes):
                if isinstance(bbox, dict):
                    rect_bbox = list(map(round, bbox['bbox']))
                    if "id" in bbox:
                        cv2.putText(img, str(bbox["id"]), (rect_bbox[0], rect_bbox[1]), cv2.FONT_HERSHEY_PLAIN, 2,
                                    colors[i], 2)
                    elif "tracking_id" in bbox:
                        cv2.putText(img, str(bbox["tracking_id"]), (rect_bbox[0], rect_bbox[1]), cv2.FONT_HERSHEY_PLAIN, 2,
                                    colors[i], 2)
                else:
                    rect_bbox = list(map(round, bbox[:4]))
                # print(bbox)
                cv2.rectangle(img, (rect_bbox[0], rect_bbox[1]), (rect_bbox[2], rect_bbox[3]), color=colors[i],
                        thickness=2)

    def display_result(self, freq_seq, results):
        img = self.get_img(freq_seq)
        # print(results)
        try:
            self.display_bboxes(img, results)
        except Exception as e:
            pass
        self.imshow(self.im.adjust_gamma(img, 2.0))

    def get_human(self, results):
        bbox = []
        for result in results:
            if int(result[-1]) == 0:
                bbox.append(result)
        return bbox
