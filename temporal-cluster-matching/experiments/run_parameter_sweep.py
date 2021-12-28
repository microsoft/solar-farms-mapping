'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import time
import itertools
import subprocess
from multiprocessing import Process, Queue

def do_work(work):
    while not work.empty():
        experiment = work.get()
        print(experiment)
        subprocess.call(experiment.split(" "))
    return True

NUM_PROCESSES = 5
work = Queue()

################################################
# Run the algorithm with the dataset footprints
################################################
datasets = ["poultry_barns", "solar_farms_reduced"]
cluster_options = {
    "poultry_barns": [16, 32, 64],
    "solar_farms_reduced": [16, 32, 64],
}
buffer_options = {
    "poultry_barns": [400,200,100],
    "solar_farms_reduced": [0.024,0.016],
}
for dataset in datasets:
    for num_clusters, buffer, in itertools.product(cluster_options[dataset], buffer_options[dataset]):
        command = f"python run_algorithm.py --dataset {dataset} --num_clusters {num_clusters} --buffer {buffer} --output_dir results/kl/{dataset}-{num_clusters}-{buffer}/ --algorithm kl"
        work.put(command)

################################################
# Run the algorithm with the random polygons
################################################
datasets = ["poultry_barns_random", "solar_farms_reduced_random"]
cluster_options = {
    "poultry_barns_random": [16, 32, 64],
    "solar_farms_reduced_random": [16, 32, 64],
}
buffer_options = {
    "poultry_barns_random": [400,200,100],
    "solar_farms_reduced_random": [0.024,0.016],
}
for dataset in datasets:
    for num_clusters, buffer, in itertools.product(cluster_options[dataset], buffer_options[dataset]):
        command = f"python run_algorithm.py --dataset {dataset} --num_clusters {num_clusters} --buffer {buffer} --output_dir results/kl/{dataset}-{num_clusters}-{buffer}/ --algorithm kl"
        work.put(command)


## Start experiments
processes = []
start_time = time.time()
for i in range(NUM_PROCESSES):
    p = Process(target=do_work, args=(work,))
    processes.append(p)
    p.start()
for p in processes:
    p.join()