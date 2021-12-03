import torch
import glob
import random
import os
import ImageTransformNet as itn
import copy
from FER2013Dataset import FER2013Dataset


class RandomStyleTransform:
    def __init__(self):
        # load style weights
        styles_path = glob.glob(".{}style_weight{}*.pth".format(os.sep, os.sep))

        self.style_weights = []
        for style_path in styles_path:
            print(style_path)
            self.style_weights.append(torch.load(style_path))

    # expected input tensor of size (BxCxHxW)
    # [0, 0.5, 1, 0.3, ...]
    def apply(self, dataset: FER2013Dataset, weights: torch.tensor):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        images = dataset.get_images().to(device)
        labels = dataset.get_labels().to(device)
        weights = weights.to(device)
        dataset_aug = copy.deepcopy(dataset)

        labels_occurrences = torch.bincount(labels, minlength=7)
        labels_to_generate = (labels_occurrences * weights).floor().cpu().numpy().astype(int)

        image_transform_net = itn.ImageTransformNet().to(device)
        n_styles = len(self.style_weights)
        for label in range(7):
            class_images = images[labels == label]
            for _ in range(0, labels_to_generate[label], 100):
                # sample of size 100
                perm = torch.randperm(class_images.size(0))
                idx = perm[:100]
                samples = class_images[idx]

                # delete extracted elements
                mask = torch.ones(class_images.size(0), dtype=torch.bool)
                mask[idx] = False
                class_images = class_images[mask]

                random_choice = random.randint(0, n_styles - 1)
                random_style_weights = self.style_weights[3]
                image_transform_net.load_state_dict(random_style_weights)
                with torch.no_grad():
                    generated_images = image_transform_net(samples)

                dataset_aug.append_images(generated_images)
                dataset_aug.append_labels(torch.ones(generated_images.size(0)) * label)

        return dataset_aug
