import torch
import glob
import random
import os
import ImageTransformNet as itn
from FER2013Dataset import FER2013Dataset


class RandomStyleTransform:
    def __init__(self):
        # load style weights
        styles_path = glob.glob(".{}style_weight{}*.pth".format(os.sep, os.sep))

        self.style_weights = []
        for style_path in styles_path:
            self.style_weights.append(torch.load(style_path))

    # expected input tensor of size (BxCxHxW)
    def apply(self, dataset: FER2013Dataset, weights):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        images = dataset.get_images().to(device)
        labels = dataset.get_labels().to(device)

        image_transform_net = itn.ImageTransformNet().to(device)
        n_styles = len(self.style_weights)
        for label, weight in enumerate(weights):
            class_images = images[labels == label]
            for class_image in class_images:
                class_image = class_image.unsqueeze(0)
                # load weights of randomly chosen style
                random_choice = random.randint(0, n_styles - 1)
                random_style_weights = self.style_weights[0]
                image_transform_net.load_state_dict(random_style_weights)

                # apply transformation
                with torch.no_grad():
                    generated_image = image_transform_net(class_image)

                dataset.append_images(generated_image)
                dataset.append_labels(torch.tensor(label).unsqueeze(0))

        return dataset
