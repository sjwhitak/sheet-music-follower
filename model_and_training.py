# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wx0Aw8lWMIUzprv9EVSzBZCEwYPhiuOk
"""

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from numpy import array, hstack
import scipy.io.wavfile as wav
import librosa
import librosa.display
import matplotlib.pyplot as plt
import json
# %matplotlib inline

MIN_MIDI = 21
TIMESTEPS = 113

def decode_one_hot_vector(vec):
  ret = []
  for pred in vec:
    note = list(pred).index(np.max(pred))
    ret.append(note+MIN_MIDI)
  return ret

def flatten(lst):
  return np.reshape(lst, len(lst))

def encode_truth(datums):
  ret = []
  for i in range(len(datums)):
    foo = np.zeros(107)
    for note in datums[i]:
      foo[note-MIN_MIDI] = 1
    ret.append(foo)
  return ret

def foo_loss(y_true, y_pred):
  return abs(y_true - y_pred)

def prep_data(data, timestep):
  ndims = data.ndim
  if ndims == 2: # Multi-dimensional
    dataOut = np.empty((data.shape[0]-timestep, timestep, data.shape[1]))
  elif ndims == 1: # 1-dimensional
    dataOut = np.empty((data.shape[0]-timestep, timestep))
  for i in range(len(data)-timestep):
    dataOut[i] = data[i:i+timestep]
  return dataOut

def load(indices, set_type=""):
  labels = []
  data = []
  for i in indices:
    with open("data/" + set_type + str(i+1) + ".json", "r") as file:
      qux = json.loads(file.read())
      labels.append(to_categorical(np.array(np.reshape(qux, len(qux))) - MIN_MIDI, num_classes=107))

    foo, sr = librosa.load("data/" + set_type + str(i+1) + ".wav", sr=16000)
    #foo = 10*np.log10(librosa.feature.melspectrogram(y=foo, sr=sr, n_fft=int(sr/10)))
    foo = librosa.feature.melspectrogram(y=foo, sr=sr, n_fft=int(sr/10))
    foo = prep_data(foo, TIMESTEPS)
    data.append(foo)
  return data, labels

def accuracy(a1, a2):
  print(a1)
  print(a2)
  return np.sum([a1[i] == a2[i] for i in range(len(a1))])/len(a1)

x_train, y_train = load(list(range(22)))
x_val, y_val = load([23])
x_test, y_test = load(list(range(15)), set_type="test")

def get_model():
  model = Sequential()
  model.add(LSTM(22, return_sequences=True, input_shape=(TIMESTEPS, 235)))
  model.add(Dropout(0.3))
  model.add(LSTM(11, return_sequences=False))
  model.add(Dropout(0.3))
  model.add(Flatten())
  model.add(Dense(107, activation="softmax"))
  model.compile(optimizer=Adam(learning_rate=0.1), loss="categorical_crossentropy")
  return model

model = get_model()
histories = []
for i in range(22):
  histories.append(model.fit(x_train[i], y_train[i], validation_data=(x_val, y_val), epochs=100, verbose=1))
  model.reset_states()

y_preds = [model.predict(x_test[i]) for i in range(15)]
spliced_preds = []
spliced_truth = []
for i in range(len(y_preds)):
  spliced_preds.extend(decode_one_hot_vector(y_preds[i]))
  spliced_truth.extend(decode_one_hot_vector(y_test[i]))

print(len(np.unique(spliced_preds)))
accuracy(spliced_preds, spliced_truth)

with open("figures/results.json", "w") as file:
  file.write(json.dumps([yhat1, ytrue1, yhat2, ytrue2]))

plt.cla(); plt.clf()
for i in range(22):
  plt.plot(histories[i].history["loss"])
plt.title("Training losses of all 22 samples")
plt.xlabel("Epoch")
plt.ylabel("Categorical Cross Entropy")
plt.savefig("figures/losses.png")

plt.cla(); plt.clf()
for i in range(22):
  plt.plot(histories[i].history["val_loss"])
plt.title("Validation losses of all 22 samples")
plt.xlabel("Epoch")
plt.ylabel("Categorical Cross Entropy")
plt.savefig("figures/validations.png")

for i in range(22):
  plt.cla(); plt.clf()
  plt.plot(histories[i].history["loss"])
  plt.plot(histories[i].history["val_loss"])
  plt.legend(["Training", "Validation"])
  plt.xlabel("Epoch")
  plt.ylabel("Categorical Cross Entropy")
  plt.title("Training sample " + str(i+1) + " training and validation")
  plt.savefig("figures/train_sample_" + str(i+1) + ".png")

time_series_val_loss = []
for i in range(22):
  time_series_val_loss.extend(histories[i].history["val_loss"])
plt.cla(); plt.clf()
plt.plot(time_series_val_loss)
plt.xlabel("Epoch")
plt.ylabel("Categorical Cross Entropy")
plt.title("Spliced validation loss over time")
plt.savefig("figures/joint_validation.png")

