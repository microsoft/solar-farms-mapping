'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import os
from multiprocessing import Process
import numpy as np


GPUS = [0, 2]  # list of IDs of the GPUs that we want to use
TEST_MODE = False  # if False then just print out the commands to be run, if True then run those commands
MODEL_FN = ''  # path passed to `--model_fn` in the `inference.py` script
OUTPUT_DIR = ''  # path passed to `--output_dir` in the `inference.py` script
IMAGERY_FILENAMES_CSV = ''  # path to a csv file containing the filenames of the imagery you want to do inference on

# -------------------
# Calculate the list of files we want our model to run on 
# -------------------
fns = []
with open(IMAGERY_FILENAMES_CSV, 'r') as f:
    for line in f:
        line = line.strip()
        if line != '':
            if line.endswith('.tif'):
                fns.append(line)


# -------------------
# Split the list of files up into approximately equal sized batches based on the number
# of GPUs we want to use. Each worker will then work on NUM_FILES / NUM_GPUS files in 
# parallel.
# Save these batches of the original list to disk (as a simple list of files to be
# consumed by the `inference.py` script) 
# -------------------

num_files = len(fns)
num_splits = len(GPUS)
num_files_per_split = np.ceil(num_files / num_splits)
output_fns = []
for split_idx in range(num_splits):
    output_fn = 'runs/2018_karnataka_predictions_final_split_%d.txt' % (split_idx)
    with open(output_fn, 'w') as f:
        start_range = int(split_idx * num_files_per_split)
        end_range = min(num_files, int((split_idx+1) * num_files_per_split))
        print('Split %d: %d files' % (split_idx+1, end_range-start_range))
        for i in range(start_range, end_range):
            end = '' if i == end_range-1 else '\n'
            f.write('%s%s' % (fns[i], end))
    output_fns.append(output_fn)

# -------------------
# Start NUM_GPUS worker processes, each pointed to one of the lists of files we saved 
# to disk in the previous step.
# -------------------


def do_work(fn, gpu_idx):
    command = f'python tile_inference.py --input_fn {fn} --model_dir {MODEL_FN} --output_dir {OUTPUT_DIR} --gpu {gpu_idx}'
    print(command)
    if not TEST_MODE:
        os.system(command)


processes = []
for work, gpu_idx in zip(output_fns, GPUS):
    p = Process(target=do_work, args=(work, gpu_idx))
    processes.append(p)
    p.start()
for p in processes:
    p.join()