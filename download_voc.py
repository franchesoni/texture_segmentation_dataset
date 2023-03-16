import os

import torchvision

DATA_DIR = '/home/franchesoni/data/tsd'

def main():
  voc = torchvision.datasets.VOCSegmentation(os.path.join(DATA_DIR, 'voc'), year='2012', image_set="train", download=True, transforms=None)

if __name__ == '__main__':
  main()
