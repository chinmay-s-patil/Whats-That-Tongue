import os
from pathlib import Path
import ffmpeg
import torch
import torchaudio
import demucs
from pydub import AudioSegment
import subprocess
import math

from demucs import pretrained
from demucs.apply import apply_model
from mir_eval import separation
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.utils import download_asset
from torchaudio.transforms import Fade

from typing import Tuple, List, Dict

from tqdm import tqdm
from IPython.display import Audio
import warnings
import json
import librosa

### Hyperparameters
SAMPLE_RATE = 44100
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

### Saving MFCC's
def extract_features(n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Take the input of the dataset and saves the JSON file to a particular folder containing MFCCs influenced by the parameters taken as input 

    Args:
        dataset_path (_type_): Preprocessed daataset path to extract the MFCCs
        json_path (_type_): JSON file path to save the MFCCs
        n_mfcc (int, optional): Number of MFCC Coefficients. Defaults to 13.
        n_fft (int, optional): Number of Fast Fourier Transform Filters. Defaults to 2048.
        hop_length (int, optional): Number of Frames to skip after the previous one. Defaults to 512.
        num_segments (int, optional): Number of segments, the audio file should split into. Defaults to 5.
    """
    # Dictionary to store the data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    
    dataset_path = r"E:\Projects\Song-Language-Classifier\WhatsThatTongue\database\chunks"
    json_path = r"E:\Projects\Song-Language-Classifier\WhatsThatTongue\database\mfcc.json"
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_second = math.ceil(num_samples_per_segment / hop_length)
    
    print(num_samples_per_segment, expected_num_mfcc_vectors_per_second)
    
    # Loop through all audio files
    for i, (dir_path, dir_names, file_names) in enumerate(os.walk(dataset_path)):
        # print(i)
        # print(dir_path, dir_names, file_names)
        
        # Process files for specific language
        for f in file_names:
            file_path = os.path.join(dir_path, f)
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            print(f)
            print(file_path)
            print(num_segments)
            
            counter = 0
            
            # Process segments, extracting mfccs and storing data
            for s in range(num_segments):
                print(s)
                start_sample = num_samples_per_segment * s
                finish_sample = start_sample + num_samples_per_segment
                
                print(start_sample, finish_sample)
                
                print(signal[start_sample:finish_sample])
                
                mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                            sr=sr,
                                            n_mfcc=n_mfcc,
                                            n_fft=n_fft,
                                            hop_length=hop_length
                                            )
                
                mfcc = mfcc.T
                
                # Store MFCC for segment if it has the expected length
                if len(mfcc) == expected_num_mfcc_vectors_per_second:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i-1)
                    print(f"{file_path}, segment:{s}")
                
                if counter == 0:
                    break
                    
                counter += 1
            break
    
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
    
    return True

chunk_path = r"WhatsThatTongue\database\chunks"


extract_features(n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5)