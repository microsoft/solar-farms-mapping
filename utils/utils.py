'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import os 
import fiona
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def get_all_geoms_from_file(fn):
    geoms = []
    with fiona.open(fn) as f:
        for row in f:
            geom = row["geometry"]
            geoms.append(geom)
    return geoms


# Methods for getting solar farm geoms
def get_solar_farm_geoms(base_dir="./data/", polygons_fn="solar_farms_India_merged_4326.geojson"):
    return get_all_geoms_from_file(os.path.join(base_dir, polygons_fn))


def scale(x, min_val, max_val, a=0, b=255, output_type=np.uint8):
    y = np.clip((x - min_val) / (max_val - min_val), 0, 1)
    y = (b-a) * y + a
    y = y.astype(output_type)
    return y