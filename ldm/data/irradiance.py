from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import CenterCrop, Compose, ToTensor

from taming.data.base import ImagePaths
import numpy as np


class IrradianceBase(ImageFolder):
    def __init__(self, config=None):
        root = config.root
        transform = Compose([CenterCrop((256, 256)), ToTensor()])
        super().__init__(root=root, transform=transform)

        paths = [path[0] for path in self.imgs]
        labels = {
            "relpath": np.array(paths),
            "synsets": np.array([0] * len(paths)),
            "class_label": np.array([0] * len(paths)),
            "human_label": np.array(["rad"] * len(paths)),
        }

        self.data = ImagePaths(paths,
                               labels=labels,
                               size=256,
                               random_crop=False,
                               )
    def __getitem__(self, i):
        return self.data[i]


class AlbedoBase(ImageFolder):
    def __init__(self, config=None):
        root = config.root
        transform = Compose([CenterCrop((256, 256)), ToTensor()])
        super().__init__(root=root, transform=transform, is_valid_file=lambda x: True)

        paths = [path[0] for path in self.imgs]
        labels = {
            "relpath": np.array(paths),
            "synsets": np.array([0] * len(paths)),
            "class_label": np.array([0] * len(paths)),
            "human_label": np.array(["albedo"] * len(paths)),
        }

        self.data = EXRPaths(paths,
                               labels=labels,
                               size=256,
                               random_crop=False,
                               )
    def __getitem__(self, i):
        return self.data[i]


class EXRPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = readEXR(image_path)
        image = image * 255
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image


def readEXR(filename):
    """Read color + depth data from EXR image file.

    Parameters
    ----------
    filename : str
        File path.

    Returns
    -------
    img : RGB or RGBA image in float32 format. Each color channel
          lies within the interval [0, 1].
          Color conversion from linear RGB to standard RGB is performed
          internally. See https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
          for more information.

    Z : Depth buffer in float32 format or None if the EXR file has no Z channel.
    """
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()

    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    # convert all channels in the image to numpy arrays
    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    if 'A' in header['channels']:
        colorChannels = ['R', 'G', 'B', 'A']
    elif 'Y' in header['channels']:
        colorChannels = ['Y']
    else:
        colorChannels = ['R', 'G', 'B']
    img = np.concatenate([channelData[c][..., np.newaxis] for c in colorChannels], axis=2)

    # # linear to standard RGB
    # img[..., :3] = np.where(img[..., :3] <= 0.0031308,
    #                         12.92 * img[..., :3],
    #                         1.055 * np.power(img[..., :3], 1 / 2.4) - 0.055)
    #
    # # sanitize image to be in range [0, 1]
    # img = np.where(img < 0.0, 0.0, np.where(img > 1.0, 1, img))

    # Z = None if 'Z' not in header['channels'] else channelData['Z']

    return img