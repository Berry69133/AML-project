import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os


class FER2013Dataset(Dataset):
    def __init__(self, dir, transforms=None):
        self.transforms = transforms
        path = './fer2013/{}'.format(dir)
        images = []
        labels = []

        for label_dir in os.listdir(path):
            label_dir_path = path + '/' + label_dir

            for image_path in os.listdir(label_dir_path):
                image_path = label_dir_path + '/' + image_path

                # features processing
                image = cv2.imread(image_path)
                image = image.astype('float32')
                image = image / 255
                image = image.swapaxes(0, 2)  # channel last to channel first

                # labels processing
                label = int(label_dir)  # directory's name is equal to the label

                images.append(image)
                labels.append(label)

        self._images = torch.from_numpy(np.array(images))
        self._labels = torch.from_numpy(np.array(labels))

    def get_labels(self):
        return self._labels

    def get_images(self):
        return self._images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self._images[idx]
        label = self._labels[idx]
        if self.transforms:
            image = self.transforms(image)
        return (image, label)

    def append_labels(self, to_append):
        self._labels = torch.cat((self._labels, to_append), dim=0)

    def append_images(self, to_append):
        self._images = torch.cat((self._images, to_append), dim=0)
