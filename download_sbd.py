import os

import torchvision

DATA_DIR = '/home/franchesoni/data/tsd'

def main():
  sbd = torchvision.datasets.SBDataset(os.path.join(DATA_DIR, 'sbd'), image_set="train", mode="segmentation", download=True, transforms=None)

if __name__ == '__main__':
  main()
