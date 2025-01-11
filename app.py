import streamlit as st
import os
from pathlib import Path
import torch
import torchaudio
import time
import math

import helper_functions as hf


def main():
    st.set_page_config(page_title="What's That Tongue?", layout="centered")
    st.title("What's That Tongue?")
    
    uploaded_file = st.file_uploader(
        "Upload your audio file",
        type=["mp3", "wav"]
    )
    
    if uploaded_file is not None:
        # File Upload Section
        upload_progress = st.progress(0)
        with st.spinner('Uploading...'):
            success = hf.save_uploaded_file(uploaded_file)
            upload_progress.progress(100)
            time.sleep(1)
        
        if success:
            st.success('Upload completed!')
            
            # Preprocessing Section
            st.subheader("Audio Preprocessing")
            
            # Convert to WAV
            with st.spinner('Converting to WAV format...'):
                try:
                    wav_progress = st.progress(0)
                    wav_placeholder = st.empty()  # Placeholder for the success message
                    wav_success = hf.convert_to_wav()
                    wav_progress.progress(100)
                    if wav_success:
                        wav_placeholder.success('WAV conversion completed!')
                        time.sleep(1)
                        wav_placeholder.empty()  # Clear the message
                    else:
                        wav_placeholder.error('WAV conversion failed.')
                except Exception as e:
                    st.error(f'An error occurred: {e}')
                finally:
                    wav_progress.empty()
            
            # Enhance Quality
            with st.spinner('Enhancing audio quality...'):
                try:
                    quality_progress = st.progress(0)
                    quality_placeholder = st.empty()  # Placeholder for the success message
                    quality_success = hf.enhance_quality()
                    quality_progress.progress(100)
                    if quality_success:
                        quality_placeholder.success('Quality enhancement completed!')
                        time.sleep(1)
                        quality_placeholder.empty()  # Clear the message
                    else:
                        quality_placeholder.error('Quality enhancement failed.')
                        return
                except Exception as e:
                    st.error(f'An error occurred: {e}')
                finally:
                    quality_progress.empty()
            
            # Remove Background Noise
            with st.spinner('Removing background noise...'):
                try:
                    noise_progress = st.progress(0)
                    noise_placeholder = st.empty()  # Placeholder for the success message
                    noise_success = hf.separate_vocals()
                    # noise_success = True
                    noise_progress.progress(100)
                    if noise_success:
                        noise_placeholder.success('Background noise removal completed!')
                        time.sleep(1)
                        noise_placeholder.empty()  # Clear the message
                    else:
                        noise_placeholder.error('Background noise removal failed.')
                        return
                except Exception as e:
                    st.error(f'An error occurred: {e}')
                finally:
                    noise_progress.empty()

            # Remove Silence
            with st.spinner('Removing silence...'):
                try:
                    silence_progress = st.progress(0)
                    silence_placeholder = st.empty()  # Placeholder for the success message
                    silence_success = hf.remove_silence()
                    silence_progress.progress(100)
                    if silence_success:
                        silence_placeholder.success('Silence removal completed!')
                        time.sleep(1)
                        silence_placeholder.empty()  # Clear the message
                    else:
                        silence_placeholder.error('Silence removal failed.')
                        return
                except Exception as e:
                    st.error(f'An error occurred: {e}')
                finally:
                    silence_progress.empty()

            # Chunking
            with st.spinner('Chunking...'):
                try:
                    chunking_progress = st.progress(0)
                    chunking_placeholder = st.empty()  # Placeholder for the success message
                    chunking_success = hf.chunk_audio(stream=True)
                    chunking_progress.progress(100)
                    if chunking_success:
                        chunking_placeholder.success('Chunking completed!')
                        time.sleep(1)
                        chunking_placeholder.empty()  # Clear the message
                    else:
                        chunking_placeholder.error('Chunking failed.')
                        return
                except Exception as e:
                    st.error(f'An error occurred: {e}')
                finally:
                    chunking_progress.empty()

            # Feature Extraction
            with st.spinner('Extracting features...'):
                try:
                    feature_progress = st.progress(0)
                    feature_placeholder = st.empty()  # Placeholder for the success message
                    feature_success = hf.extract_mfcc_from_chunks()
                    feature_progress.progress(100)
                    if feature_success:
                        feature_placeholder.success('Feature extraction completed!')
                        time.sleep(1)
                        feature_placeholder.empty()  # Clear the message
                    else:
                        feature_placeholder.error('Feature extraction failed.')
                        return
                except Exception as e:
                    st.error(f'An error occurred: {e}')
                finally:
                    feature_progress.empty()
            
            st.success('All preprocessing steps completed successfully!')
            
        else:
            st.error('Upload failed.')

if __name__ == "__main__":
    main()