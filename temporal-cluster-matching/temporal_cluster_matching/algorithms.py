'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import numpy as np
import scipy.stats

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

def calculate_change_values(images, masks, n_clusters, num_samples_for_kmeans=10000, use_minibatch=False):
    '''
    Args:
        imagery: A list of `numpy.ndarray` of shape (height, width, n_channels). This imagery should cover an area that is larger than the parcel of interest by some fixed distance (i.e. a buffer value).
        masks: A list of boolean `numpy.ndarray` of shape (height, width) with `1` in locations where the parcel covers and `0` everywhere else.
        n_clusters: The number of clusters to use in the k-means model.
        num_samples_for_kmeans: An integer specifying the number of samples to use to fit the k-means model. If `None` then all pixels in the neighborhood + footprint are used, however this is probably overkill.
        use_minibatch: A flag that indicates whether we should use MiniBatchKMeans over KMeans. MiniBatchKMeans should be much faster.

    Returns:
        divergences: A list of KL-divergence values
    '''
    divergences = []
    for image, mask in zip(images, masks):
        h,w,c = image.shape
        assert mask.shape[0] == h and mask.shape[1] == w

        mask = mask.astype(bool)

        # fit a k-means model and use it to cluster the image
        if use_minibatch:
            cluster_model = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, batch_size=2000, compute_labels=True, init="random")
        else:
            cluster_model = KMeans(n_clusters=n_clusters, n_init=3)
        features = image.reshape(h*w, c)

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        if num_samples_for_kmeans is None or (h*w <= num_samples_for_kmeans):
            labels = cluster_model.fit_predict(features)
        else:
            cluster_model.fit(features[np.random.choice(features.shape[0], size=num_samples_for_kmeans)])
            labels = cluster_model.predict(features)
        labels = labels.reshape(h,w)

        # select the cluster labels that fall within the parcel and those outside of the parcel
        parcel_labels = labels[mask]
        neighborhood_labels = labels[~mask]

        # compute the frequency with which each cluster occurs in the parcel and outside of the parcel
        parcel_counts = np.bincount(parcel_labels.ravel(), minlength=n_clusters)
        neighborhood_counts = np.bincount(neighborhood_labels.ravel(), minlength=n_clusters)

        if parcel_labels.shape[0] > 0:
            # normalize each vector of cluster index counts into discrete distributions
            parcel_distribution = (parcel_counts + 1e-5) / parcel_counts.sum()
            neighborhood_distribution = (neighborhood_counts + 1e-5) / neighborhood_counts.sum()

            # compute the KL divergence between the two distributions
            divergence = scipy.stats.entropy(parcel_distribution, neighborhood_distribution)
            divergences.append(divergence)
        else:
            divergences.append(float('inf'))

    return divergences


def calculate_change_values_with_color(images, masks):
    '''
    Args:
        imagery: A list of `numpy.ndarray` of shape (height, width, n_channels). This imagery should cover an area that is larger than the parcel of interest by some fixed distance (i.e. a buffer value).
        masks: A list of boolean `numpy.ndarray` of shape (height, width) with `1` in locations where the parcel covers and `0` everywhere else.

    Returns:
        distances: A list of Euclidean distances
    '''
    distances = []
    for image, mask in zip(images, masks):
        h,w,c = image.shape
        assert mask.shape[0] == h and mask.shape[1] == w


        colors_inside = image[mask==1].mean(axis=0)
        colors_outside = image[mask==0].mean(axis=0)

        distances.append(np.linalg.norm(
            colors_outside - colors_inside
        ))


    return distances
