import os

import torchvision

from config import DATA_DIR, DOWNLOAD_IF_NEEDED

sbd_dataset = torchvision.datasets.SBDataset(os.path.join(DATA_DIR, 'sbd'), image_set="train", mode="segmentation", download=DOWNLOAD_IF_NEEDED, transforms=None)

raise NotImplementedError("I couldn't download it because the server was down, please try again")
