import os

import imgaug.augmenters as iaa
import numpy as np
import cv2
import shutil


class MyRain(iaa.meta.SomeOf):
    """Add falling snowflakes to images.
    This is a wrapper around
    :class:`~imgaug.augmenters.weather.RainLayer`. It executes 1 to 3
    layers per image.
    .. note::
        This augmenter currently seems to work best for medium-sized images
        around ``192x256``. For smaller images, you may want to increase the
        `speed` value to e.g. ``(0.1, 0.3)``, otherwise the drops tend to
        look like snowflakes. For larger images, you may want to increase
        the `drop_size` to e.g. ``(0.10, 0.20)``.
    Added in 0.4.0.
    **Supported dtypes**:
        * ``uint8``: yes; tested
        * ``uint16``: no (1)
        * ``uint32``: no (1)
        * ``uint64``: no (1)
        * ``int8``: no (1)
        * ``int16``: no (1)
        * ``int32``: no (1)
        * ``int64``: no (1)
        * ``float16``: no (1)
        * ``float32``: no (1)
        * ``float64``: no (1)
        * ``float128``: no (1)
        * ``bool``: no (1)
        - (1) Parameters of this augmenter are optimized for the value range
              of ``uint8``. While other dtypes may be accepted, they will lead
              to images augmented in ways inappropriate for the respective
              dtype.
    Parameters
    ----------
    drop_size : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        See :class:`~imgaug.augmenters.weather.RainLayer`.
    speed : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        See :class:`~imgaug.augmenters.weather.RainLayer`.
    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.
    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.
    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.
    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.
    Examples
    --------
    """

    # Added in 0.4.0.
    def __init__(self, nb_iterations=(1, 3),
                 drop_size=(0.01, 0.02),
                 speed=(0.04, 0.20),
                 angle=-15,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        #         layer = iaa.RainLayer(
        #             density=(0.03, 0.14),
        #             density_uniformity=(0.8, 1.0),
        #             drop_size=drop_size,
        #             drop_size_uniformity=(0.2, 0.5),
        #             angle=(-15, 15),
        #             speed=speed,
        #             blur_sigma_fraction=(0.001, 0.001),
        #             seed=seed,
        #             random_state=random_state,
        #             deterministic=deterministic
        #         )

        layer = iaa.RainLayer(
            density=(0.1, 0.1),
            density_uniformity=(0.8, 0.8),
            drop_size=drop_size,
            drop_size_uniformity=(0.8, 0.8),
            angle=(angle),
            speed=speed,
            blur_sigma_fraction=(0.01, 0.01),
            seed=seed,
            random_state=random_state,
            deterministic=deterministic
        )
        super(MyRain, self).__init__(
            nb_iterations,
            children=[layer.deepcopy() for _ in range(3)],
            random_order=False,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


def rainy_augment(image_sample, bbox_file, out_of_view_file,
                  rainy_sample, rainy_bbox, rainy_outside,
                  angles=np.linspace(-40, 40, 10),
                  iterations=None):
    if iterations is None:
        iterations = [1, 2, 3]
    angle = np.random.choice(angles)
    rain = MyRain(speed=(0.2), drop_size=(0.2), angle=angle, nb_iterations=(np.random.choice(iterations)))
    aug = iaa.Sequential([rain])
    path = os.path.join(rainy_sample, f'rainy_{image_sample.split("/")[-1]}')
    if not os.path.isdir(path):
        os.mkdir(path)

    images = sorted(os.listdir(image_sample))
    for ind, image_file in enumerate(images):
        image = cv2.imread(os.path.join(image_sample, image_file))
        aug_img = aug(image=image)
        cv2.imwrite(os.path.join(path, image_file), aug_img)

    shutil.copy(bbox_file, os.path.join(rainy_bbox, f'rainy_{bbox_file.split("/")[-1]}'))
    shutil.copy(out_of_view_file, os.path.join(rainy_outside,
                                               f'rainy_{out_of_view_file.split("/")[-1]}'))


if __name__ == '__main__':
    bbox_file = "../data/bounding_boxes/005.txt"
    out_of_view_file = "../data/out_of_view/005.txt"
    image_sample = "../data/train/005"

    rainy_sample = "rainy_sample"
    rainy_bbox = "rainy_bbox"
    rainy_outside = "rainy_outside"

    rainy_augment(image_sample, bbox_file, out_of_view_file,
                  rainy_sample, rainy_bbox, rainy_outside,
                  angles=np.linspace(-40, 40, 10),
                  iterations=None)
