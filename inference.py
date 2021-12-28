'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import os
from models.unet import UnetModel
import pickle
import glob
import torch
from skimage import io
import numpy as np
from torch.autograd import Variable
from argparse import ArgumentParser
import tifffile
from skimage.transform import resize
from skimage.measure import find_contours
from skimage.draw import polygon


mean = [660.5929, 812.9481, 1080.6552, 1398.3968, 1662.5913, 1899.4804, 2061.932, 2100.2792, 2214.9325, 2230.5973, 2443.3014, 1968.1885],
std = [137.4943, 195.3494, 241.2698, 378.7495, 383.0338, 449.3187, 511.3159, 547.6335, 563.8937, 501.023, 624.041, 478.9655]


parser = ArgumentParser()
parser.add_argument(
        "--model_dir",
        type=str,
        default='/data/models/unet_hnm',
        help="Path to the directory where the solar mapping model is located at"
    )

parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        default="output/test_hnm/",
        help="Path to the directory where you wish to save predictions"
    )

parser.add_argument(
        "-r",
        "--test_dir",
        type=str,
        default="output/test_hnm/",
        help="Path to the directory where you wish to save predictions"
    )

parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2  0,2')
args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')


class InferenceFramework():
    def __init__(self, model, opts):
        self.opts = opts
        self.model = model(self.opts)
        self.model.to(device)

    def load_model(self, path_2_model):
        checkpoint = torch.load(path_2_model)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def predict_single_image(self, x):
        y_pred = self.model.forward(x.unsqueeze(0))
        return np.squeeze(np.argmax((Variable(y_pred).data).cpu().numpy(), axis=1))


def load_options(file_name):
    opt = pickle.load(open(file_name + '.pkl', 'rb'))
    return opt


def get_prediction(x, opts):
    if opts.model == "unet":
        model = UnetModel
    else:
        raise NotImplementedError

    inf_framework = InferenceFramework(
        model,
        opts
    )
    inf_framework.model.to(device)
    inf_framework.load_model(os.path.join(args.model_dir, "/checkpoint.pth.tar"))

    y_hat = inf_framework.predict_single_image(torch.from_numpy(x).float().to(device))
    return y_hat


def get_test_images(test_dir):
    assert os.path.exists(test_dir)
    all_files = []
    all_files.extend(glob.glob(os.path.join(test_dir, '*.tif')))
    return all_files


# Example
def main():
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    opts = load_options(args.model_dir + '/opt')
    test_dir = args.test_dir
    for file_name in get_test_images(test_dir):
        img = tifffile.imread(file_name)
        im = np.clip((tifffile.imread(file_name) / 3000), a_min=0, a_max=1)
        img = np.moveaxis(img, 0, 2)
        x_im = np.moveaxis(im, 0, 2)
        img = resize(img, (128, 128, 12), anti_aliasing=True)
        x_im = resize(x_im, (128, 128, 12), anti_aliasing=True)
        r, c, _ = x_im.shape
        img_rgb = np.zeros((r, c, 3))
        img_rgb[:, :, 0] = x_im[:, :, 4]
        img_rgb[:, :, 1] = x_im[:, :, 3]
        img_rgb[:, :, 2] = x_im[:, :, 2]
        x = (img-mean) / std
        x = np.moveaxis(x, 2, 0)
        y_hat = np.squeeze(get_prediction(x, opts))
        save_file = file_name.replace(test_dir, args.results_dir)
        save_file = save_file.replace(".tif", ".png")
        directory_name = os.path.dirname(save_file)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        contours = find_contours(y_hat, 0.5)
        solar_mask = np.zeros(y_hat.shape)
        for n, contour in enumerate(contours):
        # Construct the rotatedbox. If its aspect ratio is too small, we ignore it
            ll, ur = np.min(contour, 0), np.max(contour, 0)
            wh = ur - ll
            if wh[0] * wh[1] < 49:
                continue
            else:
                # fill mask for solar farm
                rr, cc = polygon(contour[:, 0], contour[:, 1], solar_mask.shape)
                solar_mask[rr, cc] = 1


if __name__ == '__main__':
    main()