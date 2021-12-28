'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import os
import time
import datetime
import argparse

from temporal_cluster_matching import utils, DataInterface, algorithms

parser = argparse.ArgumentParser(description='Script for running temporal cluster matching')
parser.add_argument('--dataset', required=True,
    choices=["poultry_barns", "solar_farms_reduced", "poultry_barns_random", "solar_farms_reduced_random"],
    help='Dataset to use'
)
parser.add_argument('--algorithm', default='kl',
    choices=(
        'kl',
        'color'
    ),
    help='Algorithm to use'
)

parser.add_argument('--num_clusters', type=int, required=False, help='Number of clusters to use in k-means step.')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--buffer', type=float, help='Amount to buffer for defining a neighborhood. Note: this will be in terms of units of the dataset.')

parser.add_argument('--output_dir', type=str, required=True, help='Path to an empty directory where outputs will be saved. This directory will be created if it does not exist.')
parser.add_argument('--verbose', action="store_true", default=False, help='Enable training with feature disentanglement')
parser.add_argument('--overwrite', action='store_true', default=False, help='Ignore checking whether the output directory has existing data')

args = parser.parse_args()


def main():
    start_time = time.time()
    print("Starting algorithm at %s" % (str(datetime.datetime.now())))

    ##############################
    # Ensure output directory exists
    ##############################
    if os.path.exists(args.output_dir):
        if not args.overwrite:
            print("WARNING: The output directory exists, exiting...")
            return
    else:
        os.makedirs(args.output_dir, exist_ok=False)

    output_fn = os.path.join(
        args.output_dir,
        "results.csv"
    )
    if os.path.exists(output_fn):
        os.remove(output_fn)

    ##############################
    # Load geoms / create dataloader
    ##############################
    if args.dataset == "poultry_barns":
        geoms = utils.get_poultry_barn_geoms()
        dataloader = DataInterface.NAIPDataLoader()
        if args.buffer is not None and args.buffer < 1:
            print("WARNING: your buffer distance is probably set incorrectly, this should be in units of meters.")

    elif args.dataset == "solar_farms_reduced":
        geoms = utils.get_solar_farm_geoms()
        dataloader = DataInterface.S2DataLoader()
        if args.buffer is not None and args.buffer > 1:
            print("WARNING: your buffer distance is probably set incorrectly, this should be in units of degrees (at equator, more/less)")

    elif args.dataset == "poultry_barns_random":
        geoms = utils.get_random_polygons_over_poultry_area()
        dataloader = DataInterface.NAIPDataLoader()
        if args.buffer is not None and args.buffer < 1:
            print("WARNING: your buffer distance is probably set incorrectly, this should be in units of degrees (at equator, more/less)")

    elif args.dataset == "solar_farms_reduced_random":
        geoms = utils.get_random_polygons_over_solar_area()
        dataloader = DataInterface.S2DataLoader()
        if args.buffer is not None and args.buffer > 1:
            print("WARNING: your buffer distance is probably set incorrectly, this should be in units of degrees (at equator, more/less)")



    ##############################
    # Loop through geoms and run
    ##############################
    tic = time.time()
    for i in range(len(geoms)):
        if i % 10 == 0:
            print("%d/%d\t%0.2f seconds" % (i, len(geoms), time.time() - tic))
            tic = time.time()


        data_images, masks, years = dataloader.get_data_stack_from_geom(geoms[i], buffer=args.buffer)

        if args.algorithm == "kl":
            divergence_values = algorithms.calculate_change_values(data_images, masks, n_clusters=args.num_clusters)
        elif args.algorithm == "color":
            divergence_values = algorithms.calculate_change_values_with_color(data_images, masks)

        with open(output_fn, "a") as f:
            f.write("%d," % (i))
            for year in years:
                f.write("%d," % (year))
            f.write("|,")
            for divergence in divergence_values:
                f.write("%0.4f," % (divergence))
            f.write("\n")


    print("Finished in %0.2f seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()