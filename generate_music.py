#!/usr/bin/env python3
from load_data import single_data_loader
from spectra import pitch_to_frequency
import random
import numpy as np
import json
from scipy.io import wavfile as wav

def unique_list_indices(x):
    # Take a list with multiple values, then re-align for each duplicate.
    # Example:
    # >>> x = [0, 0, 1, 2, 3, 9, 4, 6, 7, 6]
    # >>> unique_list_indices(x)
    # ([0, 1, 2, 3, 4, 6, 7, 9],
    # [[0, 1], [2], [3], [4], [6, 8], [7], [5]])
    
    x_uniq = list(set(x))
    y = list()
    for uniq in x_uniq:
        y_uniq = list()
        for i,e in enumerate(x):
            if e == uniq:
                y_uniq.append(i)
        if y_uniq:
            y.append(y_uniq)
    return x_uniq, y

def choose_unique_list(Y, N):
    # Y is output of unique_list_indices, so
    # >>> x_val, x_ind = unique_list_indices(x)
    # >>> y_val, y_ind = __choose_notes(x_ind, 2) // Choosing 2 in the list
    Y_rand = random.choices(list(enumerate(Y)), k=N)
    Y_values = [y[1] for y in Y_rand] # values
    Y_indices = [y[0] for y in Y_rand]    # indices
    return Y_values, Y_indices


# NOTE(sjwhitak): I call this a wrapper because it's not really a library
# function, rather just a massive bulk function that does everything in
# one go. I want to keep chopping this function up.
def generate_song_wrapper(X, Y, N, note_range=[0,2], bpm=120, fs=16000):
    Y_values, Y_indices = unique_list_indices(Y)
    # For N beats,
    note_audio_list = list()
    midi_list = list()
    note_per_beat_list = list()
    beats = list()
    for n in range(N):
        note_audio = np.zeros(X[0].shape)
        
        # Choose how many notes are played in this beat
        note_count = 1#random.randrange(note_range[0], note_range[1])
        if note_count != 0:
    
            # Choose notes at random
            midi_values, midi_indices = choose_unique_list(Y_indices, note_count)
            midi_list.append(midi_values) # values
            
            # Choose random note inside this list of all these pitches
            for notes in midi_values:
                r = random.choices(notes, k=1)[0]
                note_audio += X[r]
            note_per_beat_list.append([Y_values[x] for x in midi_indices])
        else:
            note_per_beat_list.append([0])
            pass # Rest note
        
        # Determine length of note (half note, quarter note, eight note, etc)
        note_speeds = [0.25, 0.5, 1, 2] # bpm is based of quarter notes
        speed = 1#random.choice(note_speeds)
        
        # Cut down note based on BPM
        audio_len = note_audio.shape[0]
        quarter_note_size = bpm/60 *audio_len/fs
        size = quarter_note_size*speed
        note_audio = note_audio[:int(audio_len/size)]
        
        # End of beat, compile the note.
        note_audio_list.append(note_audio)
        beats.append(speed)
    
    # Create a "song" of random notes.
    song = np.hstack(note_audio_list)    
    return song, note_per_beat_list, beats



if __name__ == "__main__":
    dataset_path = 'dataset/'
    subset ='keyboard_acoustic' 
    
    # Generate training data
    dataset_folder = 'train/'
    X, Y = single_data_loader(dataset_path, dataset_folder, subset)
    song, values, length = generate_song_wrapper(X, Y, 4000, [0,3])
    wav.write("train.wav", 16000, song.astype(np.float32))
    np.save("train_labels.npy", np.array(values))
    print("Training data generated")
    
    # Generate validation data
    dataset_folder = 'valid/'
    X, Y = single_data_loader(dataset_path, dataset_folder, subset)
    song, values, length = generate_song_wrapper(X, Y, 400, [0,3])
    wav.write("valid.wav", 16000, song.astype(np.float32))
    np.save("valid_labels.npy", np.array(values))
    print("Validation data generated")
        
    # Generate test data
    dataset_folder = 'test/'
    X, Y = single_data_loader(dataset_path, dataset_folder, subset)
    song, values, length = generate_song_wrapper(X, Y, 400, [0,3])
    wav.write("test.wav", 16000, song.astype(np.float32))
    np.save("test_labels.npy", np.array(values))
    print("Test data generated")