import enum
import cv2
import glob
import time

import torch

import utils
import ImageTransformNet

def stylize(style_choice, content_img_path, style_transform_path):
    # Check the style weight path
    style_filepath = {}
    style_name_list = []
    weight_path = glob.glob("./style_weight/*.pth")
    number_of_style = len(weight_path)
    for path in weight_path:
        weight_name = path.split("/")[-1]
        style_name = weight_name.replace(".pth", "")
        style_name_list.append(style_name)
        style_filepath[style_name] = weight_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference_device = "GPU" if device == "cuda" else "CPU"
    print(f'Now you use "{inference_device}" to processing Fast Style Transfer')

    transform_net = transform_net.ImageTransformNet()
    with torch.no_grad():
        content_image = utils.load_image(content_img_path)

        starttime = time.time()

        content_tensor = utils.img2tensor(content_image).to(device)

        transform_net.load_state_dict(torch.load(style_transform_path))
        transform_net = transform_net.to(device)

        # Conver image to new style
        content_tensor = utils.img2tensor(content_image).to(device)
        generated_tensor = net(content_tensor)
        generated_image = utils.tensor2img(generated_tensor.detach())
        print("Transfer Time: {}".format(time.time() - starttime))
        utils.show(generated_image)