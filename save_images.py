import pandas as pd
import numpy as np
import argparse
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices = {"train", "test"} , help='choose train or test.')
opt = parser.parse_args()

data_split = opt.dataset

labels = pd.read_csv("./data/labels_{}.csv".format(data_split))

list_to_save = labels.i.unique()

X_train = np.load('./data/data_{}.npy'.format(data_split), mmap_mode='r')

output_dir = "images_{}".format(data_split)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
for i in list_to_save:
    
    output_file = os.path.join(output_dir, 'image_{}.jpg'.format(i))
    
    cv2.imwrite(output_file, X_train[i])