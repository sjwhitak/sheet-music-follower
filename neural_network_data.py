#!/usr/bin/env python3

from load_data import single_data_loader
from generate_music import generate_song_wrapper

dataset_path = 'dataset/'
dataset_folder = 'test/'
subset ='keyboard_acoustic' 

X, Y = single_data_loader(dataset_path, dataset_folder, subset)
song, values, length = generate_song_wrapper(X, Y, 30, [0,6])