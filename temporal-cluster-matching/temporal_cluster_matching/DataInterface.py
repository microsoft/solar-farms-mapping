'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import abc
from functools import lru_cache

import numpy as np

from skimage.segmentation import mark_boundaries

import rasterio
import rasterio.mask
import rasterio.features
import rasterio.windows
import rasterio.warp

import shapely
import shapely.geometry

from pystac_client import Client
import planetary_computer as pc

from . import utils

# Some tricks to make rasterio faster when using vsicurl -- see https://github.com/pangeo-data/cog-best-practices
RASTERIO_BEST_PRACTICES = dict(
    CURL_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt',
    GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
    AWS_NO_SIGN_REQUEST='YES',
    GDAL_MAX_RAW_BLOCK_CACHE_SIZE='200000000',
    GDAL_SWATH_SIZE='200000000',
    VSI_CURL_CACHE_SIZE='200000000'
)

def get_mask_and_bounding_geoms(geom, buffer):
    '''Returns the two polygons needed to crop imagery with given a query geometry and buffer amount.
    The Temporal Cluster Matching algorithm will cluster all pixels in a footprint + neighborhood, then form distribution of cluster indices from the pixels within a footprint and a distribution with the pixels in the neighborhood.
    To calculate this, we need to crop the imagery from the entire buffered extent and know which of those pixels fall within the footprint. The two polyongs we return here let us do that.

    Args:
        geom: A polygon in GeoJSON format describing the query footprint.
        buffer: An amount (in units of `geom`'s coordinate system) to buffer the geom by.

    Returns:
        mask_geom: A polygon in GeoJSON format that has the same extent as `bounding_geom`, but has a hole where `geom` is.
        bounding_geom: A polygon in GeoJSON format that is the extent of `geom` after being buffered by `buffer`.
    '''
    footprint_shape = shapely.geometry.shape(geom).buffer(0.0)
    bounding_shape = footprint_shape.envelope.buffer(buffer).envelope
    mask_geom = shapely.geometry.mapping(bounding_shape - footprint_shape) # full bounding area - initial footprint
    bounding_geom = shapely.geometry.mapping(bounding_shape) # full bounding area
    return mask_geom, bounding_geom


################################################################
################################################################
class AbstractDataLoader(abc.ABC):
    ''' This class facilitates loading patches of imagery from a source time-series of remotely sensed imagery in a way that can be used by the Temporal Cluster Matching algorithm.
    '''

    @abc.abstractmethod
    def get_rgb_stack_from_geom(self, geom, buffer, show_outline=True):
        """Returns a time-series stack of RGB image patches corresponding to a query geometry (that optionally show the outline of the query geometry).

        Args:
            geom: A polygon in GeoJSON format describing the query footprint.
            buffer: An amount (in units of imagery's projection) to buffer the geom by.
            show_outline: A flag that indicates whether the RGB image patches should be rendered with the outline of `geom`.

        Returns:
            rgb_images: A list of RGB image patches (with `np.uint8` dtypes), one for each date in the source time-series. Each patch should be a crop that covers the extent of the `geom` buffered by an amount specified by `buffer`.
            dates: A list of dates corresponding to each patch in `images`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data_stack_from_geom(self, geom, buffer):
        """Returns a time-series stack of data images corresponding to a query geometry. While `get_rgb_stack_from_geom(.)` returns just the RGB component of the imagery, this method should return
        the bands to be included in processing.

        Args:
            geom: A polygon in GeoJSON format describing the query footprint.
            buffer: An amount (in units of imagery's projection) to buffer the geom by.
        Returns:
            images: A list of image patches (with a `dtype` matching the source time-series), one for each date in the source time-series. Each patch should be a crop that covers the extent of the `geom` buffered by an amount specified by `buffer`.
            masks: A list of masks for each patch in `images`. These should be binary, contain a 1 where the corresponding image is covered by the `geom`, and contain a 0 elsewhere.
            dates: A list of dates corresponding to each patch in `images`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def data_stack_to_rgb(self, images):
        """A convenience method that converts the `images` that are returned by `get_data_stack_from_geom(.)` to `rgb_images` (i.e. the kind returned by `get_rgb_stack_from_geom`).
        This is its own method because if you have `images` from `get_data_stack_from_geom(.)` already, it is likely cheaper to reprocess those into `rgb_images` instead of hitting your data source to re-download the
        RGB components of your data.

        Args:
            images: The list of image patches that are returned by `get_data_stack_from_geom(.)`.
        Returns:
            rgb_images: A list of RGB image patches (with `np.uint8` dtypes), one for each patch in `images`. These should be processed in the same way that `get_rgb_stack_from_geom(.)` processes the source imagery.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dates_from_geom(self, geom):
        """A convenience method for determining what dates of data are available for a given geometry.

        Args:
            geom: A polygon in GeoJSON format describing the query footprint.
        Returns:
            dates: A list of dates for which there is corresponding data for `geom`.
        """
        raise NotImplementedError()


################################################################
################################################################
class NAIPDataLoader(AbstractDataLoader):

    def __init__(self):
        self.index = utils.NAIPTileIndex()

    def _get_fns_from_geom(self, geom):

        centroid = utils.get_transformed_centroid_from_geom(geom, src_crs='epsg:26918', dst_crs='epsg:4326')
        fns = self.index.lookup_tile(*centroid)
        fns = sorted(fns)

        base_state = fns[0].split("/")[1]

        valid_fns = []
        years = []
        for fn in fns:

            year = int(fn.split("/")[2])
            state = fn.split("/")[1]

            if year in years:
                continue
            if state != base_state:
                continue

            valid_fns.append(fn)
            years.append(int(year))

        valid_fns = np.array(valid_fns)
        years = np.array(years)

        idxs = np.argsort(years)
        valid_fns = valid_fns[idxs]

        return valid_fns

    def get_dates_from_geom(self, geom):
        fns = self._get_fns_from_geom(geom)

        years = []
        for fn in fns:
            year = int(fn.split("/")[2])
            years.append(year)
        return years

    def get_rgb_stack_from_geom(self, geom, buffer, show_outline=True):

        mask_geom, bounding_geom = get_mask_and_bounding_geoms(geom, buffer)
        fns = self._get_fns_from_geom(geom)

        years = []
        images = []
        for fn in fns:

            year = int(fn.split("/")[2])
            years.append(year)

            with rasterio.Env(**RASTERIO_BEST_PRACTICES):
                with rasterio.open(utils.NAIP_BLOB_ROOT + fn) as f:
                    mask_image, _ = rasterio.mask.mask(f, [mask_geom], crop=True, invert=False, pad=False, all_touched=True)
                    mask_image = np.rollaxis(mask_image, 0, 3)

                    full_image, _ = rasterio.mask.mask(f, [bounding_geom], crop=True, invert=False, pad=False, all_touched=True)
                    full_image = np.rollaxis(full_image, 0, 3)[:,:,:3]

                    mask = np.zeros((mask_image.shape[0], mask_image.shape[1]), dtype=np.uint8)
                    mask[np.sum(mask_image == 0, axis=2) != 4] = 1

            if show_outline:
                images.append(mark_boundaries(
                    full_image, mask
                ))
            else:
                images.append(full_image)

        return images, years

    def get_data_stack_from_geom(self, geom, buffer):

        mask_geom, bounding_geom = get_mask_and_bounding_geoms(geom, buffer)
        fns = self._get_fns_from_geom(geom)

        years = []
        images = []
        masks = []
        for fn in fns:

            year = int(fn.split("/")[2])
            years.append(year)

            with rasterio.Env(**RASTERIO_BEST_PRACTICES):
                with rasterio.open(utils.NAIP_BLOB_ROOT + fn) as f:
                    mask_image, _ = rasterio.mask.mask(f, [mask_geom], crop=True, invert=False, pad=False, all_touched=True)
                    mask_image = np.rollaxis(mask_image, 0, 3)

                    full_image, _ = rasterio.mask.mask(f, [bounding_geom], crop=True, invert=False, pad=False, all_touched=True)
                    full_image = np.rollaxis(full_image, 0, 3)

                    mask = np.zeros((mask_image.shape[0], mask_image.shape[1]), dtype=np.bool)
                    mask[np.sum(mask_image==0, axis=2) == 4] = 1

            images.append(full_image)
            masks.append(mask)

        return images, masks, years

    def data_stack_to_rgb(self, images):
        rgb_images = []
        for image in images:
            rgb_images.append(image[:,:,:3])
        return rgb_images


################################################################
################################################################
class S2DataLoader(AbstractDataLoader):

    years = [
        2016, 2017, 2018, 2019, 2020
    ]
    urls = [
        "https://researchlabwuopendata.blob.core.windows.net/sentinel-2-imagery/karnataka_change/2016/2016_merged.tif",
        "https://researchlabwuopendata.blob.core.windows.net/sentinel-2-imagery/karnataka_change/2017/2017_merged.tif",
        "https://researchlabwuopendata.blob.core.windows.net/sentinel-2-imagery/karnataka_change/2018/2018_merged.tif",
        "https://researchlabwuopendata.blob.core.windows.net/sentinel-2-imagery/karnataka_change/2019/2019_merged.tif",
        "https://researchlabwuopendata.blob.core.windows.net/sentinel-2-imagery/karnataka_change/2020/2020_merged.tif",
    ]

    def get_dates_from_geom(self, geom):
        return list(S2DataLoader.years)

    def get_rgb_stack_from_geom(self, geom, buffer, show_outline=True):

        mask_geom, bounding_geom = get_mask_and_bounding_geoms(geom, buffer)

        years = list(S2DataLoader.years)
        images = []
        for url in S2DataLoader.urls:

            with rasterio.Env(**RASTERIO_BEST_PRACTICES):
                with rasterio.open(url) as f:
                    mask_image, _ = rasterio.mask.mask(f, [mask_geom], crop=True, invert=False, pad=False, all_touched=True)
                    mask_image = np.rollaxis(mask_image, 0, 3)
                    mask_image = mask_image[:,:,[3,2,1]]

                    full_image, _ = rasterio.mask.mask(f, [bounding_geom], crop=True, invert=False, pad=False, all_touched=True)
                    full_image = np.rollaxis(full_image, 0, 3)
                    full_image = full_image[:,:,[3,2,1]]
                    full_image = utils.scale(1.1*full_image, 0, 2500)

                    mask = np.zeros((mask_image.shape[0], mask_image.shape[1]), dtype=np.uint8)
                    mask[np.sum(mask_image == 0, axis=2) != 3] = 1

            if show_outline:
                images.append(mark_boundaries(
                    full_image, mask
                ))
            else:
                images.append(full_image)

        return images, years

    def get_data_stack_from_geom(self, geom, buffer):

        mask_geom, bounding_geom = get_mask_and_bounding_geoms(geom, buffer)

        years = list(S2DataLoader.years)
        images = []
        masks = []
        for url in S2DataLoader.urls:

            with rasterio.Env(**RASTERIO_BEST_PRACTICES):
                with rasterio.open(url) as f:
                    mask_image, _ = rasterio.mask.mask(f, [mask_geom], crop=True, invert=False, pad=False, all_touched=True)
                    mask_image = np.rollaxis(mask_image, 0, 3)
                    mask_image = mask_image[:,:,[3,2,1]]

                    full_image, _ = rasterio.mask.mask(f, [bounding_geom], crop=True, invert=False, pad=False, all_touched=True)
                    full_image = np.rollaxis(full_image, 0, 3)
                    full_image = full_image[:,:,[1,2,3,7]] # keep B, G, R, and NIR bands

                    mask = np.zeros((mask_image.shape[0], mask_image.shape[1]), dtype=np.bool)
                    mask[np.sum(mask_image == 0, axis=2) == 3] = 1

            images.append(full_image)
            masks.append(mask)

        return images, masks, years

    def data_stack_to_rgb(self, images):
        rgb_images = []
        for image in images:
            image = image[:,:,[2,1,0]]
            image = utils.scale(1.1*image, 0, 2500)
            rgb_images.append(image)
        return rgb_images


################################################################
################################################################


class PlanetaryComputerS2DataLoader(AbstractDataLoader):

    def __init__(self, geoms, pc_subscription_key, search_start="2015-01-01", search_end="2019-12-31"):
        pc.settings.set_subscription_key(pc_subscription_key)
        self.geoms = geoms
        self.time_range = f"{search_start}/{search_end}"

    @lru_cache(maxsize=None)
    def query_geom(self, geom_idx):
        geom = self.geoms[geom_idx]
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=geom,
            datetime=self.time_range,
            query={"eo:cloud_cover": {"lt": 10}},
        )

        items = list(search.get_items())
        return items[::-1]

    def get_dates_from_geom(self, geom_idx):
        items = self.query_geom(geom_idx)
        dates = []
        for item in items:
            dates.append(item.datetime.strftime("%m-%d-%Y"))
        return dates

    def get_rgb_stack_from_geom(self, geom_idx, buffer, show_outline=True):

        images, masks, dates = self.get_data_stack_from_geom(geom_idx, buffer)
        if show_outline:
            new_images = []
            for image, mask in zip(images, masks):
                new_images.append(mark_boundaries(
                    image, mask
                ))
            return new_images, dates
        else:
            return images, dates

    @lru_cache(maxsize=None)
    def get_data_stack_from_geom(self, geom_idx, buffer):
        geom = self.geoms[geom_idx]

        items = self.query_geom(geom_idx)
        dates = self.get_dates_from_geom(geom_idx)

        crss = set()
        for item in items:
            crss.add(item.properties["proj:epsg"])
        assert len(crss) == 1
        dst_crs = "epsg:" + str(list(crss)[0])

        geom = rasterio.warp.transform_geom("epsg:4326", dst_crs, geom)
        mask_geom, bounding_geom = get_mask_and_bounding_geoms(geom, buffer)

        images = []
        masks = []
        for item in items:

            href = item.assets["visual-10m"].href
            signed_href = pc.sign(href)

            with rasterio.Env(**RASTERIO_BEST_PRACTICES):
                with rasterio.open(signed_href) as f:

                    mask_image, _ = rasterio.mask.mask(f, [mask_geom], crop=True, invert=False, pad=False, all_touched=True)
                    mask_image = np.rollaxis(mask_image, 0, 3)

                    full_image, _ = rasterio.mask.mask(f, [bounding_geom], crop=True, invert=False, pad=False, all_touched=True)
                    full_image = np.rollaxis(full_image, 0, 3)

                    mask = np.zeros((mask_image.shape[0], mask_image.shape[1]), dtype=np.uint8)
                    mask[np.sum(mask_image == 0, axis=2) != 3] = 1

            images.append(full_image)
            masks.append(mask)

        return images, masks, dates

    def data_stack_to_rgb(self, images):
        raise NotImplementedError("This method is unecessary as the data is already RGB")
