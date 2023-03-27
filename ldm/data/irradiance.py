from torchvision.datasets import ImageFolder
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
            "human_label": np.array([0] * len(paths)),
        }

        self.data = ImagePaths(paths,
                               labels=labels,
                               size=256,
                               random_crop=False,
                               )
    def __getitem__(self, i):
        return self.data[i]

