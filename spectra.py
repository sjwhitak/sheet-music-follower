#!/usr/bin/env python3
import load_data as ld
import numpy as np
import librosa


def pitch_to_frequency(pitch):
    """
    Parameters
    ----------
    pitch : List of ints or int of MIDI tune value.
    
    Returns
    -------
    f : List of float32s or float32 of frequency (Hz) corresp. to MIDI pitch.
    """
    # https://en.wikipedia.org/wiki/MIDI_tuning_standard
    if isinstance(pitch, list):
        f = [440 * 2 ** ((x - 69)/12) \
             for x in pitch]
    elif isinstance(pitch, int):
        f =  400 * 2 ** ((pitch - 69)/12)
    else:
        # You should only use this as a list or a single value.
        # If you somehow get here, then you're not using this properly.
        raise NotImplementedError
    return f

def wav_to_mel(audio_stream, n_mels, fs):
    """
    Parameters
    ----------
    audio_stream : list of np.arrays or np.array of audio
    n_mels : Count of mel-spectrum frequency bins.
    fs : float32 Sample frequency
    
    Returns
    -------
    S : Mel-spectrum amplitude in [frequency, time] in linear power.
    f : Mel-spectrum frequency as 1-D array
    t : Time as 1-D array
    """
    if isinstance(audio_stream, list):
        S = [librosa.feature.melspectrogram(x, sr=fs, n_mels=n_mels) \
                   for x in audio_stream]
        t = np.linspace(0, audio_stream[0].shape[0]/fs, S[0].shape[1])
    elif isinstance(audio_stream, np.ndarray):
        S = librosa.feature.melspectrogram(audio_stream, sr=fs, n_mels=n_mels)
        t = np.linspace(0, audio_stream.shape[0]/fs, S[0].shape[1])
    else:
        # You should only use this as a list or a single array.
        # If you somehow get here, then you're not using this properly.
        raise NotImplementedError
    f = librosa.convert.mel_frequencies(fmin=1, fmax=fs/2, n_mels=n_mels)
    return S, f, t

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    pylab.rcParams.update({'legend.fontsize': 'xx-large',
              'figure.figsize': (15, 10),
             'axes.labelsize': 'xx-large',
             'axes.titlesize':24,
             'xtick.labelsize':'xx-large',
             'ytick.labelsize':'xx-large'})
    
    # Load all the data
    dataset_path = 'dataset/'
    dataset_folder = 'test/'
    subset ='keyboard_acoustic' 
    X,Y = ld.single_data_loader(dataset_path, dataset_folder, subset)
    
    subsample = 1 # Choose one value from this list
    fs = 16000    # Sample frequency, can be found in json but it's all the same.
    n_mels = 128  # Mel-frequency spacing.
    time = X[0].shape[0]/fs # Time for each audio file (will be 4 seconds)
    
    Y_hz = pitch_to_frequency(Y)
    S, f, t = wav_to_mel(X, n_mels, fs)
    
    # NOTE(sjwhitak): Somehow this properly plots the frequencies.
    # When I try writing my own plotting routine, the frequencies are
    # poorly setup. Literally doesn't matter, but I'd like to know how
    # for my own research.
    plt.figure()
    librosa.display.specshow(10*np.log10(S[subsample]), x_axis='time',
                          y_axis='mel', sr=fs,
                          fmax=8000, cmap='magma')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Pitch: ' + str(np.round(Y_hz[subsample], 3)) + ' Hz')
    plt.savefig('out.png')
    plt.close()