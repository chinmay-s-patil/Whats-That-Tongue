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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')
warnings.filterwarnings('ignore', category=FutureWarning, module='torchaudio')

def save_uploaded_file(uploaded_file):
    """
    Save the uploaded file to the uploads directory
    """
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Create file path
        file_path = "uploads/test.mp3"
        
        if os.path.exists(file_path): os.remove(file_path)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        return True
    except Exception as e:
        print(e)
        return False

def convert_to_wav():
    
    input_path = r"uploads/test.mp3"
    output_path = r"uploads/test.wav"
    replace = True
    
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"The file '{input_path}' does not exist.")
    
    # Handle WAV file replacement logic
    if os.path.isfile(output_path):
        if replace:
            os.remove(output_path)
        else:
            raise FileExistsError(f"The file '{output_path}' already exists. Set 'replace' to True to overwrite.")
    
    # Perform the conversion
    try:
        ffmpeg.input(input_path).output(output_path).run(overwrite_output=True)
        print(f"Conversion successful: {output_path}")
        
        #os.remove(input_path)
        
        return True
    except ffmpeg.Error as e:
        raise RuntimeError(f"Error during conversion: {e.stderr.decode()}")

def enhance_quality():
    """
    Optimize MP3 file with high quality settings.
    
    Args:
        input_path: Path to input MP3 file
        output_path: Optional output path. If None, overwrites input file
        
    Returns:
        Path to optimized file
    """
    input_path = r"uploads/test.wav"
    output_path = r"uploads/test_enhanced.wav"
    
    try:
        stream = ffmpeg.input(str(input_path))
        stream = ffmpeg.output(
            stream,
            str(output_path),
            acodec='libmp3lame',
            ab='320k',    # Highest common bitrate
            ar='44100',   # High quality sample rate
            ac=2,         # Stereo
            loglevel='error'
        )
        ffmpeg.run(stream, overwrite_output=True)
        
        #os.remove(input_path)
        
        return True
    except ffmpeg.Error as e:
        raise RuntimeError(f"Error during conversion: {e.stderr.decode()}")

bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model().to(device)
sample_rate = bundle.sample_rate

def separate_vocals(model=model,
                   segment=10.0,
                   overlap=0.1,
                   device=device):
    """
    Load an audio file, separate the vocals, and save the result.
    Uses a fixed input path 'uploads/test_enhanced.wav' and output path 'uploads/test_vocals.wav'.
    
    Args:
        model (torch.nn.Module): Model to separate the tracks
        segment (int): segment length in seconds
        device (torch.device): device on which to execute the computation
    """
    # Load the audio file
    print("Loading audio file...")
    
    input_path = r"uploads/test_enhanced.wav"
    output_path = r"uploads/test_vocals.wav"
    
    mix, sr = torchaudio.load(input_path)
    
    # Resample if necessary
    if sr != sample_rate:
        print(f"Resampling from {sr} to {sample_rate}...")
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        mix = resampler(mix)
    
    # Add batch dimension if not present
    if mix.dim() == 2:
        mix = mix.unsqueeze(0)
    
    batch, channels, length = mix.shape
    
    print("Separating the sources...")
    
    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = sample_rate * overlap
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")
    
    final = torch.zeros(batch, len(model.sources), channels, length, device=device)
    
    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        
        with torch.inference_mode():
            out = model.forward(chunk.to(device))
        out = fade(out)
        final[:, :, :, start:end] += out
        
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end > length:
            fade.fade_out_len = 0
    
    # Extract vocals and save
    vocals = final[:, 3, :, :][0]
    print("Saving vocals to file...")
    torchaudio.save(output_path, vocals.cpu(), sample_rate)
    print("Done!")
    
    #os.remove(input_path)
    
    return True


def remove_silence():
    input_path = r"uploads/test_vocals.wav"
    output_path = r"uploads/test_sil_rem.wav"
    
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, af="silenceremove=1:0:-20dB")
            .run(overwrite_output=True)
        )
        print(f"Silence removed successfully. Output saved to {output_path}")
        
        #os.remove(input_path)
        
        return True
    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode()}")
        
        return False


def chunk_audio(window_size=30*44100, hop_length=5*44100, stream=False):
    """Chunks the audio file at upload/test_sil_rem.wav and saves chunks to database folder.
    
    Args:
        window_size (int): Length of each chunk in samples
        hop_length (int): Stride between chunks in samples
    """
    
    if not stream:
        input_path = r"uploads\test_sil_rem.wav"
        output_dir = r"database\chunks\\"
    else:
        input_path = r"uploads\test_sil_rem.wav"
        Path("database").mkdir(exist_ok=True)
        output_dir = r"database\chunks\\"
    
    print("Does: " +  input_path + " exist? " + str(os.path.exists(input_path)))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load audio file
    wave_tensor, sample_rate = torchaudio.load(input_path)
    
    # Process on device
    wave_tensor = wave_tensor.to(device)
    num_frames = wave_tensor.shape[-1]
    
    # Calculate chunks
    num_chunks = math.floor(((num_frames - window_size) / hop_length) + 1)
    start_idx = 0
    end_idx = window_size
    
    # Create and save chunks
    for i in range(num_chunks):
        chunk = wave_tensor[:, start_idx:end_idx]
        output_path = f"{output_dir}/chunk_{i:03d}.wav"
        torchaudio.save(output_path, chunk.cpu(), sample_rate)
        
        start_idx += hop_length
        end_idx += hop_length
    
    print(f"Created {num_chunks} chunks in the database folder")
    
    return True

#

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
    
    dataset_path = r"database\chunks"
    json_path = r"database\mfcc.json"
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_second = math.ceil(num_samples_per_segment / hop_length)
    
    # Loop through all audio files
    for i, (dir_path, dir_names, file_names) in enumerate(os.walk(dataset_path)):
        
        # Ensure we're not at the root level
        if dir_path is not dataset_path:
            # Save the semantic label
            semantic_label = dir_path.split("/")[-1]
            data["mapping"].append(semantic_label)
            print(f"\nProcessing {semantic_label}")
            
            # Process files for specific language
            for f in file_names:
                file_path = os.path.join(dir_path, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Process segments, extracting mfccs and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment
                    
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
                        # print(f"{file_path}, segment:{s}")
    
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
    
    return True

#
def extract_mfcc_from_chunks(n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract MFCC features from audio chunks in the 'database/chunks' folder and save them to a JSON file.

    Args:
        json_path (str): Path to save the JSON file containing MFCCs.
        n_mfcc (int, optional): Number of MFCC coefficients to extract. Defaults to 13.
        n_fft (int, optional): Number of FFT components. Defaults to 2048.
        hop_length (int, optional): Hop length for overlapping frames. Defaults to 512.
    """
    # Dictionary to store the extracted MFCCs
    data = {
        "mfcc": []
    }
    
    json_path = r"database\mfcc.json"
    
    if os.path.exists(json_path):
        os.remove(json_path)

    # Path to the chunks folder
    chunks_path = r"database\chunks"

    # Iterate over all chunk files in the folder
    for file_name in sorted(os.listdir(chunks_path)):
        file_path = os.path.join(chunks_path, file_name)

        # Check if the file is a valid audio file
        if file_name.endswith(".wav"):
            try:
                # Load the audio chunk
                signal, sr = librosa.load(file_path, sr=None)

                # Extract MFCCs
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

                # Transpose MFCC to have time steps along rows
                mfcc = mfcc.T

                # Append MFCCs to the data dictionary
                data["mfcc"].append(mfcc.tolist())

                print(f"Processed {file_name}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Save the extracted MFCCs to the JSON file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    print(f"MFCCs saved to {json_path}")
    
    return True

# 

# END