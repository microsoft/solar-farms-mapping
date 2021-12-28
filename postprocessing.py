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
import numpy as np
import glob
from skimage.measure import find_contours
from skimage.draw import polygon


parser = argparse.ArgumentParser(description='Solar Installations mapping post-processing script')

parser.add_argument('--model_predictions', type=str, help='Path to a where \
                                        predictions from model 1 are stored')
parser.add_argument('--output_dir', type=str,  help='Path to a directory where \
                    outputs will be saved. This directory will be created if it does \
                    not exist.')

args = parser.parse_args()


def main():
    print("Starting postprocessing script at %s" % (str(datetime.datetime.now())))

    # -------------------
    # Load files
    # -------------------
    assert os.path.exists(args.model_predictions)

    # Ensure output directory exists
    if os.path.exists(args.output_dir):
        if len(os.listdir(args.output_dir)) > 0:
            print("WARNING: The output directory is not empty, but we are ignoring that and writing data.") 
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    fns = []
    fns.extend(glob.glob(os.path.join(args.model_predictions + '/predictions', '*.tif')))
    print("Running on %d files" % (len(fns)))

    # -------------------
    # Load files and post-process them
    # -------------------
    for fn_idx, fn in enumerate(fns):
        model2_fn = os.path.join(args.model_predictions + '/predictions3/',
                                os.path.basename(fn))

        fn_parts = fn.split("/")
        new_fn = fn_parts[-1][:-4] + "_postprocessed.tif"

        # Read data model 1
        with rasterio.open(os.path.join(args.model_predictions, fn)) as f:
            data1 = f.read()
            data1 = np.squeeze(data1)

        # Read data model 2
        with rasterio.open(model2_fn) as f2:
            data2 = f2.read()
            data2 = np.squeeze(data2)
            profile = f2.profile
            height, width = data2.shape
        
        #Read image tile
        
        tile_fn = os.path.join(args.model_predictions + '/tiles/', str(os.path.basename(fn)).replace('_predictions.tif', '.tif'))
        with rasterio.open(tile_fn) as f:
            tile = f.read()
            img = np.moveaxis(tile, 0, 2)

        output = np.zeros((height, width), dtype=np.uint8)

        # Post process predictions 
        output[(data1 == 1) & (data2 == 1)] = 1
        print("got here")

        # Water index
        NDWI = (img[:, :, 2] - img[:, :, 7]) / (img[:, :, 2] + img[:, :, 7] + 0.0001)
        print(np.max(NDWI))
        print(np.mean(NDWI))
        # Remove solar panels on water
        output[NDWI > 30] = 0

        # remove predictions over clouds or snow
        output[img[:, :, 0] > 1300] = 0
        
        contours = find_contours(output, 0.5)
        for n, contour in enumerate(contours):
            # Construct the rotatedbox. If its aspect ratio is too small, we ignore it 
            ll, ur = np.min(contour, 0), np.max(contour, 0)
            wh = ur - ll
            if (wh[0] * wh[1] > 49):
                continue
            else:
                # Zero out small polygons
                rr, cc = polygon(contour[:, 0], contour[:, 1], output.shape)
                output[rr, cc] = 0

        # output[output==1]=0

        # Save post-processed predictions
        new_profile = profile.copy()
        new_profile["count"] = 1
        new_profile["dtype"] = "uint8"
        new_profile["compress"] = "lzw"

        with rasterio.open(os.path.join(args.output_dir, new_fn), "w", **new_profile) as f:
            f.write(output, 1)
            f.write_colormap(1, {
                0: (24, 154, 211, 0),
                1: (255, 211, 0, 255),
            })


if __name__ == "__main__":
    main()