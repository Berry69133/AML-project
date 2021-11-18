import torch
import glob
import random
import sys

import ImageTransformNet as itn


class RandomStyleTransform:
    def __init__(self):
        # load style weights
        if sys.platform == 'linux':
            styles_path = glob.glob("./style_weight/*.pth")
        else:
            styles_path = glob.glob(".\style_weight\*.pth")

        self.style_weights = []
        for style_path in styles_path:
            self.style_weights.append(torch.load(style_path))

    # expected input tensor of size (BxCxHxW)
    def apply(self, images):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        images = images.to(device)

        # load weights of randomly chosen style
        image_transform_net = itn.ImageTransformNet().to(device)
        n_styles = len(self.style_weights)
        print(n_styles)
        random_choice = random.randint(0, n_styles-1)
        random_style_weights = self.style_weights[random_choice]
        image_transform_net.load_state_dict(random_style_weights)

        # apply transformation
        with torch.no_grad():
            generated_images = image_transform_net(images)

        return generated_images
