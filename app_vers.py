import streamlit as st
import os
from pathlib import Path
import torch
import torchaudio
import helper_functions as hf
import time

def main():
    st.set_page_config(page_title="What's That Tongue?", layout="centered")
    st.title("What's That Tongue?")
    
    uploaded_file = st.file_uploader(
        "Upload your audio file",
        type=["mp3", "wav"]
    )
    
    # Create a placeholder for processing messages
    status_container = st.empty()
    progress_container = st.empty()
    
    if uploaded_file is not None:
        # File Upload Section
        with progress_container:
            upload_progress = st.progress(0)
        with status_container:
            st.spinner('Uploading...')
        
        success = hf.save_uploaded_file(uploaded_file)
        upload_progress.progress(100)
        with status_container:
            st.success('Upload completed!')
        time.sleep(1)
        
        if success:
            
            # Preprocessing Section
            st.subheader("Audio Preprocessing")
            
            # Clear previous progress
            progress_container.empty()
            status_container.empty()
            
            # Convert to WAV
            with progress_container:
                wav_progress = st.progress(0)
            with status_container:
                st.spinner('Converting to WAV format...')
            
            wav_success = hf.convert_to_wav()
            wav_progress.progress(100)
            with status_container:
                st.success('WAV conversion completed!')
            time.sleep(1)
            
            if not wav_success:
                st.error('WAV conversion failed.')
                return
                
            # Clear previous progress
            progress_container.empty()
            status_container.empty()
            
            # Enhance Quality
            with progress_container:
                quality_progress = st.progress(0)
            with status_container:
                st.spinner('Enhancing audio quality...')
            
            quality_success = hf.enhance_quality()
            quality_progress.progress(100)
            with status_container:
                st.success('Quality enhancement completed!')
            time.sleep(1)
            
            if not quality_success:
                st.error('Quality enhancement failed.')
                return
                
            # Clear previous progress
            progress_container.empty()
            status_container.empty()
            
            # Remove Background Noise
            with progress_container:
                noise_progress = st.progress(0)
            with status_container:
                st.spinner('Removing background noise...')
            
            noise_success = hf.separate_vocals()
            noise_progress.progress(100)
            with status_container:
                st.success('Background noise removal completed!')
            time.sleep(1)
            
            if not noise_success:
                st.error('Background noise removal failed.')
                return
                
            # Clear previous progress
            progress_container.empty()
            status_container.empty()
            
            # Remove Silence
            with progress_container:
                silence_progress = st.progress(0)
            with status_container:
                st.spinner('Removing silence...')
            
            silence_success = hf.remove_silence()
            silence_progress.progress(100)
            with status_container:
                st.success('Silence removal completed!')
            time.sleep(1)
            
            if not silence_success:
                st.error('Silence removal failed.')
                return
            
            # Clear final progress
            progress_container.empty()
            status_container.empty()
            
            st.success('All preprocessing steps completed successfully!')
            
        else:
            st.error('Upload failed.')

if __name__ == "__main__":
    main()