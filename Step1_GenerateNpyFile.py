#!/usr/bin/python3.7
# -*- coding: UTF-8 -*-
import numpy as np
import os
import random
from datetime import datetime as DT
import sys

#input gene expression profile
#(Y-label: gene name; X-label: sample id; content: float numeric value)
path = sys.argv[1]
dirs = os.listdir(path)

# npy output name
output_file_name=sys.argv[2]

#Set random seed to shuffle order of input files
#(avoid CNN model classify samples by file orders)
time = DT.utcnow()
seed = str(time.year) + str(time.month) + str(time.day) + str(time.hour) + str(time.minute) + str(time.microsecond)
random.seed(seed)
random.shuffle(dirs)
print(seed)

# generating npy files
sample_titles = np.array(dirs)
samples = []
labels = []
elements = []
O_names = []

for name in dirs:
    O_names.append(name)
    data = name.split('-')
    labels.append(data[2]) # 12 number for 11 cancer types + 1 normal type
    elements_name = []
    with open ( path+"/"+name ) as file:
        for line in file:
            line = line.strip().split()
            elements_name.append(line)
    elements.append(elements_name)

x_samples = np.array(elements).astype(np.float32)
y_labels = np.array(labels).astype(np.int32)
np.save(output_file_name + ".npy", x_samples) # npy file
np.save(output_file_name + "_label.npy", y_labels) # truth labels of samples
np.save(output_file_name + "_title.npy", sample_titles) # unique IDs of samples

print(x_samples.shape, y_labels.shape)
