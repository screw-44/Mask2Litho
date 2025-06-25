import numpy as np
from glob import glob
from tqdm import tqdm
import cv2
import os


labels = "./90/label/*"
sems = "./90/sem/*"
label_dirs = sorted(glob(labels))
sem_dirs = sorted(glob(sems))

paired_dirs = [(i, j) for i, j in zip(label_dirs, sem_dirs)]
for label_dir, sem_dir in tqdm(paired_dirs):
    label = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)
    bin_label = cv2.threshold(label, 120, 255, cv2.THRESH_BINARY)[1]

    bright_pixels = np.sum(bin_label > 0)
    total_pixels = bin_label.size
    ratio = bright_pixels / total_pixels
    # print(ratio)
    # print(label_dir, sem_dir)
    if ratio < 0.1:
        os.remove(label_dir)
        os.remove(sem_dir)






