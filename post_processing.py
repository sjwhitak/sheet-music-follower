#!/usr/bin/env python3
import numpy as np
import librosa
from tensorflow.keras.utils import to_categorical
MIN_MIDI = 21
TIMESTEPS = 13
FFT_SIZE = 256
MEL_SIZE = 128

qux = np.load('train_labels.npy')
train_labels = to_categorical(np.array(qux)-MIN_MIDI, num_classes=107)    
train_sound, sr = librosa.load("train.wav", sr=16000)
train_sound_stft = librosa.stft(train_sound, n_fft=FFT_SIZE)[:,:-1]
mel_filter = librosa.filters.mel(sr, FFT_SIZE, n_mels=MEL_SIZE)
train_sound_mel = np.dot(mel_filter, train_sound_stft)
del train_sound, train_sound_stft, mel_filter

qux = np.load('test_labels.npy')
test_labels = to_categorical(np.array(qux)-MIN_MIDI, num_classes=107)    
test_sound, sr = librosa.load("test.wav", sr=16000)
test_sound_stft = librosa.stft(test_sound, n_fft=FFT_SIZE)[:,:-1]
mel_filter = librosa.filters.mel(sr, FFT_SIZE, n_mels=MEL_SIZE)
test_sound_mel = np.dot(mel_filter, test_sound_stft)
del test_sound, test_sound_stft, mel_filter

train_x = np.array(np.split(train_sound_mel, train_labels.shape[0], axis=1))
test_x = np.array(np.split(test_sound_mel, test_labels.shape[0], axis=1))
train_y = train_labels
test_y = test_labels
del train_sound_mel, test_sound_mel

np.save("test_x", test_x)
np.save("test_y", test_y)
np.save("train_x", train_x)
np.save("train_y", train_y)