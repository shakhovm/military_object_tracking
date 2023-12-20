import numpy as np
import cv2
# import matplotlib.pyplot as plt


class ImagePreprocessor:
        
    # def histograms_equalization(self, img):
    #     hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    #     cdf = hist.cumsum()
    #     cdf_normalized = cdf * float(hist.max()) / cdf.max()
    #     plt.plot(cdf_normalized, color='b')
    #     plt.hist(img.flatten(), 256, [0, 256], color='r')
    #     plt.xlim([0, 256])
    #     plt.legend(('cdf', 'histogram'), loc='upper left')
    #     plt.show()

    def clahe(self, img, gridsize):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        return rgb

    def contrast(self, img, alpha, beta):
        return alpha*img + beta

    def adjust_gamma(self, img, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(img, table)

    def blur(self, img, kernel=(5, 5)):
        return cv2.blur(img, kernel)

    def gaussian_blur(self, img, kernel=(5, 5), depths=0):
        return cv2.GaussianBlur(img, kernel, depths
                                )

    def bilaterial_filter(self, img):
        return cv2.bilateralFilter(img, 9, 75, 75)

    def to_rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)