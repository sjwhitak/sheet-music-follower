#!/usr/bin/env python3
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams.update({'legend.fontsize': 'xx-large',
          'figure.figsize': (15, 10),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':24,
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'})
MIN_MIDI = 21
TIMESTEPS = 128
FFT_SIZE = 256
MEL_SIZE = 128

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

def normalize(x, r=None):
    if r is None:
        return (x - np.min(x))/(np.max(x) - np.min(x))
    else:
        return (x - r[0])/(r[1] - r[0])


def get_model():
  
  model = Sequential()
  model.add(Flatten(input_shape=(MEL_SIZE,80)))
  model.add(Dense(107, activation="softmax"))
  model.summary()
  model.compile(optimizer=Adam(learning_rate=0.1), loss="categorical_crossentropy")
  return model

train_x = np.load("train_x.npy")
train_y = np.load("train_y.npy")
test_x = np.load("test_x.npy")
test_y = np.load("test_y.npy")
valid_x = np.load("valid_x.npy")
valid_y = np.load("valid_y.npy")

total_max = np.max( [np.max(train_x),np.max(test_x),np.max(valid_x)] )
total_min = np.min( [np.min(train_x),np.min(test_x),np.min(valid_x)] )

train_x = normalize(train_x, [total_min, total_max])
valid_x = normalize(valid_x, [total_min, total_max])
test_x = normalize(test_x, [total_min, total_max])

model = get_model()
checkpoint = ModelCheckpoint('out.h5', 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             save_weights_only=True,
                             mode='min',
                             save_freq="epoch")
history = model.fit(train_x, train_y, 
                    validation_data=(valid_x, valid_y), epochs=150, 
                    verbose=1, callbacks=[checkpoint])

model.load_weights('out.h5')
y_pred = model.predict(test_x)
y_pred_out = np.array(decode_one_hot_vector(y_pred))
y_test_out = np.array(decode_one_hot_vector(test_y))

accuracy = np.sum(y_pred_out == y_test_out)/y_pred_out.shape[0]


