#!/usr/bin/env python3
import json
from scipy.io import wavfile as wav


def data_generator(files, batch):
    """
    Parameters
    ----------
    files : List of filenames to load
    batch : Amount of files returned

    Returns
    -------
    None.
    """
    pass

def data_files(dataset_path, dataset_folder, subset):
    """
    Parameters
    ----------
    dataset_path : (Str) path of dataset folder
    dataset_folder : (Str) "test", "train", "valid"
    subset : (Str) Class of instrument used.
        https://magenta.tensorflow.org/datasets/nsynth

    Returns
    -------
    files : List of filenames
    """
    # Open dataset values
    with open(dataset_path+'nsynth-'+dataset_folder+'examples.json','r') as fd:
        json_data = json.loads(fd.read())
    
    # Trim the dataset down using only the specified subset
    json_data = {key:val for key, val in json_data.items() if subset in key}
    
    # Open up specific audio files
    audio_files = [dataset_path+'nsynth-'+
                   dataset_folder+'audio/'+json_data[data]['note_str']+
                   '.wav' for data in json_data]
    
    return audio_files  

def single_data_loader(dataset_path, dataset_folder, subset):
    """
    Parameters
    ----------
    dataset_path : (Str) path of dataset folder
    dataset_folder : (Str) "test", "train", "valid"
    subset : (Str) Class of instrument used.
        https://magenta.tensorflow.org/datasets/nsynth

    Returns
    -------
    X : Audio data.
    Y : Audio labels.

    """
    # Open dataset values
    with open(dataset_path+'nsynth-'+dataset_folder+'examples.json','r') as fd:
        json_data = json.loads(fd.read())
    
    # Trim the dataset down using only the specified subset
    json_data = {key:val for key, val in json_data.items() if subset in key}
    
    # Open up specific audio files
    audio_files = [dataset_path+'nsynth-'+
                   dataset_folder+'audio/'+json_data[data]['note_str']+
                   '.wav' for data in json_data]
    
    # TODO(sjwhitak): This loads ALL the data at once. Convert to data loader
    # when working with training data, else it will be very slow.
    # Once data_generator() works, will change to WARN.
    audio_data = [wav.read(file)[1]/32768.0 for file in audio_files]
    pitch = [json_data[data]['pitch'] for data in json_data]
    
    # NOTE(sjwhitak): These are parameters on the distortions on the audio.
    # Maybe we want to have this as input?
    qualities = [json_data[data]['qualities'] for data in json_data]
    
    Y = pitch
    X = audio_data
    return X, Y

if __name__ == "__main__":
    
    # Data structure is:
    # dataset_path/
    # +---- nsynth-test/
    # |   \---- examples.json
    # |   \---- audio/
    # |       \---- file.wav (4096 files)
    # |       \---- ...
    # +---- nsynth-train/
    # |   \---- examples.json
    # |   \---- audio/
    # |       \---- file.wav (289205 files)
    # |       \---- ...
    # \---- nsynth-valid/
    #     \---- examples.json
    #     \---- audio/
    #         \---- file.wav (12678 files)
    #         \---- ...
    dataset_path = 'dataset/'
    dataset_folder = 'test/'
    
    # NOTE(sjwhitak): We're only doing the piano, so this removes, 
    # like 90% of the other data in our dataset.
    # If you want the whole data set, remove the dict comprehension line.
    subset ='keyboard_acoustic' 
    
    X, Y = single_data_loader(dataset_path, dataset_folder, subset)
    
    