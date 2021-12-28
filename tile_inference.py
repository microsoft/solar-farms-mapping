'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import os
import time
import datetime
import argparse
import rasterio
import rasterio.mask
import torch
import torch.nn.functional as F
import numpy as np
from models.unet import UnetModel
import pickle
from skimage.measure import find_contours
from skimage.draw import polygon
from skimage.exposure import match_histograms
from rasterio.windows import from_bounds

NAIP_BLOB_ROOT = "/"
MATCHING_IMAGERY_FN = ""
parser = argparse.ArgumentParser(description='Solar Installations mapping inference script')

parser.add_argument('--input_fn', type=str, required=True, help='Path to a text file containing a list of files to run the model on. If these do not begin with `NAIP_BLOB_ROOT` then it will be appended.')
parser.add_argument('--model_dir', type=str, default='/data/models/unet_hnm/', help='Path to the model file to use.')
parser.add_argument('--output_dir', type=str, required=True,  help='Path to a directory where outputs will be saved. This directory will be created if it does not exist.')
parser.add_argument('--gpu',  type=int, default=0,  help='ID of the GPU to run on.')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

means = [660.5929, 812.9481, 1080.6552, 1398.3968, 1662.5913, 1899.4804, 2061.932, 2100.2792, 2214.9325, 2230.5973, 2443.3014, 1968.1885],
stds = [137.4943, 195.3494, 241.2698, 378.7495, 383.0338, 449.3187, 511.3159, 547.6335, 563.8937, 501.023, 624.041, 478.9655]


def run_model_on_tile(tile, model, device, batch_size=256, use_softmax=False):
    model.eval()
    
    height, width, _ = tile.shape
    num_output_channels = 2

    input_size = 256
    down_weight_padding = 10
    stride_x = input_size - down_weight_padding*2
    stride_y = input_size - down_weight_padding*2
    output = np.zeros((height, width, num_output_channels), dtype=np.float32)

    counts = np.zeros((height, width), dtype=np.float32) + 0.000000001
    kernel = np.ones((input_size, input_size), dtype=np.float32) * 0.1
    kernel[10:-10, 10:-10] = 1
    kernel[down_weight_padding:down_weight_padding+stride_y,
           down_weight_padding:down_weight_padding+stride_x] = 5

    batches = []
    batch_indices = []
    batch_count = 0

    for y_index in (list(range(0, height - input_size, stride_y)) + [height - input_size,]):
        for x_index in (list(range(0, width - input_size, stride_x)) + [width - input_size,]):
            img = tile[y_index:y_index+input_size, x_index:x_index+input_size, :].copy()
            img = (img - means) / stds
            img = np.rollaxis(img, 2, 0).astype(np.float32)
            batches.append(img)
            batch_indices.append((y_index, x_index))
            batch_count += 1
    batches = np.array(batches)

    model_output = []
    for i in range(0, batch_count, batch_size):
        batch = torch.from_numpy(batches[i:i+batch_size])
        batch = batch.to(device)
        with torch.no_grad():
            if use_softmax:
                outputs = F.softmax(model(batch), dim=1)
            else:
                outputs = model(batch)
        outputs = outputs.cpu().numpy()
        outputs = np.rollaxis(outputs, 1, 4)
        model_output.append(outputs)
    model_output = np.concatenate(model_output, axis=0)
    for i, (y, x) in enumerate(batch_indices):
        output[y:y+input_size, x:x+input_size] += model_output[i] * kernel[..., np.newaxis]
        counts[y:y+input_size, x:x+input_size] += kernel
    output = output / counts[..., np.newaxis]
    return output


def load_options(file_name):
    opt = pickle.load(open(file_name + '.pkl', 'rb'))
    return opt


def main():
    print("Starting inference script at %s" % (str(datetime.datetime.now())))

    # Load files
    assert os.path.exists(args.input_fn)

    ## Ensure output directory exists
    if os.path.exists(args.output_dir):
        if len(os.listdir(args.output_dir)) > 0:
            print("WARNING: The output directory is not empty, but we are ignoring that and writing data.") 
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    fns = []
    with open(args.input_fn, "r") as f:
        lines = f.read().strip().split("\n")
        for line in lines:
            if not line.startswith(NAIP_BLOB_ROOT):
                fns.append(NAIP_BLOB_ROOT + "/" + line)
            else:
                fns.append(line)
    print("Running on %d files" % (len(fns)))

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    opts = load_options(args.model_dir +'/opt')
    model = UnetModel(opts)

    path_2_model = os.path.join(args.model_dir, "/checkpoint.pth.tar")
    model.to(device)
    checkpoint = torch.load(path_2_model)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Run model on all files and save output
    tic = float(time.time())
    for fn_idx, fn in enumerate(fns):
        if fn_idx % 100 == 0:
            print("%d/%d\t%0.2f seconds" % (fn_idx, len(fns), time.time()-tic))
            tic = float(time.time())

        fn_parts = fn.split("/")
        new_fn = fn_parts[-1][:-4] + "_predictions.tif"

        # Read data
        with rasterio.open(fn) as f:
            data = np.rollaxis(f.read(), 0, 3)
            boundbox = f.bounds
            left, bottom, right, top = boundbox[0], boundbox[1], boundbox[2], boundbox[3]
            profile = f.profile
            height, width, num_channels = data.shape
            # Check for Top of Atmosphere S2
            if num_channels == 13:
                source = np.zeros((height, width, 12))
                source[:, :, :10] = data[:, :, :10]
                source[:, :, 10:] = data[:, :, 11:]
                src = rasterio.open(MATCHING_IMAGERY_FN)
                reference = np.rollaxis(src.read(window=from_bounds(left, bottom, right, top, src.transform)), 0, 3)
                data = match_histograms(source, reference, multichannel=True)


        # Run inference
        output = run_model_on_tile(data, model, device)
        output = output.argmax(axis=2).astype(np.uint8)
        matches = (data == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all(axis=2)*1
        output[matches] = 2
        print("got here")

        #  Water index
        NDWI = (data[:, :, 2] - data[:, :, 7]) / (data[:, :, 2] + data[:, :, 7] + 0.0001)
        # Remove solar panels on water
        output[NDWI>25] = 0

        # remove predictions over clouds or snow
        output[data[:, :, 0] > 1200] = 0

        # Post process predictions 
        contours = find_contours(output, 0.5)
        for n, contour in enumerate(contours):
            # Construct the rotatedbox. If its aspect ratio is too small, we ignore it 
            ll, ur = np.min(contour, 0), np.max(contour, 0)
            wh = ur - ll
            if wh[0] * wh[1] > 47:
                continue
            else:
                #Zero out small polygons
                rr, cc = polygon(contour[:, 0], contour[:, 1], output.shape)
                output[rr, cc] = 0


        # Save new predictions
        new_profile = profile.copy()
        new_profile["count"] = 1
        new_profile["dtype"] = "uint8"
        new_profile["compress"] = "lzw"

        with rasterio.open(os.path.join(args.output_dir, new_fn), "w", **new_profile) as f:
            f.write(output, 1)
            f.write_colormap(1, {
                0: (24, 154, 211, 0),
                1: (255, 211, 0, 255),
                2: (1, 1, 1, 255),
            })


if __name__ == "__main__":
    main()