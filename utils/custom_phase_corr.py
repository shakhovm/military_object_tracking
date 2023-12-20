import numpy as np
import matplotlib.pyplot as plt


def cros_pow_spect(img1, img2, visualize=False):
    f = np.fft.fft2(img1)
    fshift1 = np.fft.fftshift(f)
    f = np.fft.fft2(img2)
    fshift2 = np.fft.fftshift(f)
    cross_power_spectrum = np.conj(fshift2) * fshift1

    f_ishift = cross_power_spectrum / np.abs(cross_power_spectrum)
    cps = np.fft.fftshift(np.fft.ifft2(f_ishift))
    if visualize:
        plt.imshow(np.log(np.abs(cross_power_spectrum)))
        plt.imshow(np.abs(cps))  # .real)
        plt.show()
    center = cps.shape[0] // 2, cps.shape[1] // 2
    shift = np.unravel_index(np.argmax(cps), cps.shape)
    return shift[0] - center[0], shift[1] - center[1]
