'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import os
import tempfile
import pickle
import urllib
import urllib.request

import numpy as np
import matplotlib.pyplot as plt

import rtree
import fiona
import fiona.transform
import shapely
import shapely.geometry

from sklearn.metrics import accuracy_score, mean_absolute_error

NAIP_BLOB_ROOT = 'https://naipblobs.blob.core.windows.net/naip/'

################################################################
# Dataset methods
################################################################
def get_all_geoms_from_file(fn):
    geoms = []
    with fiona.open(fn) as f:
        for row in f:
            geom = row["geometry"]
            geoms.append(geom)
    return geoms

## Methods for getting poultry barn geoms
def get_poultry_barn_geoms(base_dir="./data/"):
    return get_all_geoms_from_file(os.path.join(base_dir, "Delmarva_PL_House_Final2_epsg26918.geojson"))

def get_random_polygons_over_poultry_area(base_dir="./data/"):
    return get_all_geoms_from_file(os.path.join(base_dir, "poultry_barn_6013_random_polygons_epsg26918.geojson"))

def get_poultry_barn_geoms_epsg4326(base_dir="./data/"):
    return get_all_geoms_from_file(os.path.join(base_dir, "Delmarva_PL_House_Final2_epsg4326.geojson"))

## Methods for getting solar farm geoms
def get_solar_farm_geoms(base_dir="./data/"):
    return get_all_geoms_from_file(os.path.join(base_dir, "karnataka_predictions_polygons_validated_2020.geojson"))

def get_random_polygons_over_solar_area(base_dir="./data/"):
    return get_all_geoms_from_file(os.path.join(base_dir, "solar_farms_935_random_polygons_epsg4326.geojson"))

def get_labels(fn):
    idxs = []
    years = []
    with open(fn, "r") as f:
        lines = f.read().strip().split("\n")
        for i in range(1,len(lines)):
            parts = lines[i].split(",")
            idxs.append(int(parts[0]))
            years.append(int(parts[1]))
    return idxs, years

def get_poultry_barn_labels(base_dir="./data/"):
    return get_labels(os.path.join(base_dir, "poultry_barn_labels.csv"))

def get_solar_farm_labels(base_dir="./data/"):
    return get_labels(os.path.join(base_dir, "solar_farm_labels.csv"))

################################################################
# Visualization methods
################################################################
def show_images(images, titles=None):
    num_images = len(images)
    if titles is not None:
        assert len(titles) == num_images

    fig, axs = plt.subplots(1, num_images, figsize=(num_images*4, 4))
    axs = axs.flatten()
    for i in range(num_images):

        axs[i].imshow(images[i])
        if titles is not None:
            axs[i].set_title(titles[i])
        axs[i].axis("off")
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)

    plt.show()
    plt.close()

def show_individual_images(images, border_size=1):
    for img in images:
        h,w,c = img.shape

        if img.dtype==np.uint8:
            img = img / 255.0

        img_with_border = np.zeros((h+int(2*border_size),w+int(2*border_size),c), dtype=np.float32)
        img_with_border[border_size:-border_size,border_size:-border_size] = img

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(img_with_border)
        plt.show()
        plt.close()


def scale(x, min_val, max_val, a=0, b=255, output_type=np.uint8):
    y = np.clip((x - min_val) / (max_val - min_val), 0, 1)
    y = (b-a) * y + a
    y = y.astype(output_type)
    return y


################################################################
# Geometric methods
################################################################
def get_transformed_centroid_from_geom(geom, src_crs='epsg:26918', dst_crs='epsg:4326'):
    shape = shapely.geometry.shape(geom)
    x = shape.centroid.x
    y = shape.centroid.y
    lat, lon = fiona.transform.transform(src_crs, dst_crs, xs=[x], ys=[y])
    lat = lat[0]
    lon = lon[0]

    return (lat, lon)

def reverse_polygon_coordinates(geom):
    new_coords = []

    if geom["type"] == "MultiPolygon":
        for polygon in geom["coordinates"]:
            new_polygon = []
            for ring in polygon:
                new_ring = []
                for x, y in ring:
                    new_ring.append((y,x))
                new_polygon.append(new_ring)
            new_coords.append(new_polygon)
    elif geom["type"] == "Polygon":
        if len(geom["coordinates"][0][0]) == 2:
            for ring in geom["coordinates"]:
                new_ring = []
                for x, y in ring:
                    new_ring.append((y,x))
                new_coords.append(new_ring)
        else:
            for ring in geom["coordinates"]:
                new_ring = []
                for x, y, z in ring:
                    new_ring.append((y,x,z))
                new_coords.append(new_ring)
    geom["coordinates"] = new_coords
    return geom

def bounds_to_geom(bounds, src_crs, dst_crs):
    left, right = bounds.left, bounds.right
    top, bottom = bounds.top, bounds.bottom

    geom = {
        "type": "Polygon",
        "coordinates": [[(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]]
    }
    return fiona.transform.transform_geom(src_crs, dst_crs, geom)


################################################################
# Tile index for NAIP data
################################################################
class NAIPTileIndex:
    """
    Utility class for performing NAIP tile lookups by location.
    """
    index_blob_root = 'https://naipblobs.blob.core.windows.net/naip-index/rtree/'
    index_fns = ["tile_index.dat", "tile_index.idx", "tiles.p"]

    def __init__(self, base_path=None):

        if base_path is None:
            base_path = tempfile.gettempdir()

        for file_path in NAIPTileIndex.index_fns:
            download_url(NAIPTileIndex.index_blob_root + file_path, base_path)

        self.base_path = base_path
        self.tile_rtree = rtree.index.Index(base_path + "/tile_index")
        self.tile_index = pickle.load(open(base_path  + "/tiles.p", "rb"))


    def lookup_tile(self, lat, lon):
        """"
        Given a lat/lon coordinate pair, return the list of NAIP tiles that contain
        that location.

        Returns an array containing [mrf filename, idx filename, lrc filename].
        """

        point = shapely.geometry.Point(float(lon), float(lat))
        intersected_indices = list(self.tile_rtree.intersection(point.bounds))

        intersected_files = []
        tile_intersection = False

        for idx in intersected_indices:
            intersected_file = self.tile_index[idx][0]
            intersected_geom = self.tile_index[idx][1]
            if intersected_geom.contains(point):
                tile_intersection = True
                intersected_files.append(intersected_file)

        if not tile_intersection and len(intersected_indices) > 0:
            print('''Error: there are overlaps with tile index, but no tile completely contains selection''')
            return None
        elif len(intersected_files) <= 0:
            print("No tile intersections")
            return None
        else:
            return intersected_files


def download_url(url, output_dir, force_download=False, verbose=False):
    """
    Download a URL
    """

    parsed_url = urllib.parse.urlparse(url)
    url_as_filename = os.path.basename(parsed_url.path)
    destination_filename = os.path.join(output_dir, url_as_filename)

    if (not force_download) and (os.path.isfile(destination_filename)):
        if verbose: print('Bypassing download of already-downloaded file {}'.format(os.path.basename(url)))
        return destination_filename

    if verbose: print('Downloading file {} to {}'.format(os.path.basename(url),destination_filename),end='')
    urllib.request.urlretrieve(url, destination_filename)  
    assert(os.path.isfile(destination_filename))
    nBytes = os.path.getsize(destination_filename)
    if verbose: print('...done, {} bytes.'.format(nBytes))
    return destination_filename


################################################################
# Other stuff for processing results
################################################################
def decision_function(all_distances, all_years, theta, max_year):
    predicted_years = []
    for distances, years in zip(all_distances, all_years):
        made_prediction = False
        for distance, year in zip(distances, years):
            if distance >= theta or year >= max_year:
                predicted_years.append(year)
                made_prediction = True
                break
        if not made_prediction:
            predicted_years.append(years[-1])
    return predicted_years

def get_results(fn, filter_years=None):
    all_idxs = []
    all_years = []
    all_distances = []

    with open(fn) as f:
        lines = f.read().strip().split("\n")

        for line in lines:
            years = []
            distances = []
            parts = line.strip().strip(",").split(",")
            idx = int(parts[0])

            parts = parts[1:]

            j = 0
            while parts[j] != "|":
                years.append(int(parts[j]))
                j += 1

            j += 1
            while j < len(parts):
                distance = float(parts[j])
                if np.isnan(distance):
                    distances.append(float('inf'))
                else:
                    distances.append(distance)
                j += 1

            all_idxs.append(idx)
            if filter_years is None: 
                all_years.append(years)
                all_distances.append(distances)
            else:
                t_years = []
                t_distances = []

                for distance, year in zip(distances, years):
                    if year in filter_years:
                        t_years.append(year)
                        t_distances.append(distance)
                all_years.append(t_years)
                all_distances.append(t_distances)

    return all_idxs, all_years, all_distances


def uncertain_accuracy(labeled_years, predicted_years):
    labeled_years = np.array(labeled_years)
    predicted_years = np.array(predicted_years)
    mask = labeled_years != -1
    return accuracy_score(labeled_years[mask], predicted_years[mask])

def uncertain_mae(labeled_years, predicted_years):
    labeled_years = np.array(labeled_years)
    predicted_years = np.array(predicted_years)
    mask = labeled_years != -1
    return mean_absolute_error(labeled_years[mask], predicted_years[mask])

def loss_function(labeled_idxs, labeled_years, all_distances, all_years, theta, max_year):

    distances, years = [], []
    for idx in labeled_idxs:
        distances.append(all_distances[idx])
        years.append(all_years[idx])

    predicted_years = decision_function(distances, years, theta, max_year)

    acc = uncertain_accuracy(labeled_years, predicted_years)
    mae = uncertain_mae(labeled_years, predicted_years)

    return acc, mae