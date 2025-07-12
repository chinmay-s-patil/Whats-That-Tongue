import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
from pytube import YouTube
from pydub import AudioSegment


def download_audio_as_mp3(youtube_url):
    # Set local FFmpeg path (no admin required)
    ffmpeg_path = r"D:\ffmpeg\ffmpeg.exe"  # Your ffmpeg.exe path
    AudioSegment.converter = ffmpeg_path

    # Output folder
    folder = 'downloadedMusic'
    if not os.path.exists(folder):
        os.makedirs(folder)

    try:
        yt = YouTube(youtube_url)
        print(f"Downloading: {yt.title}")

        # Best quality audio stream
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        if not audio_stream:
            raise Exception("No audio stream found.")

        # Download
        downloaded_file = audio_stream.download(output_path=folder)
        print(f"Downloaded to: {downloaded_file}")

        # Convert to mp3
        base, ext = os.path.splitext(downloaded_file)
        mp3_file = base + '.mp3'

        if ext.lower() != '.mp3':
            audio = AudioSegment.from_file(downloaded_file)
            audio.export(mp3_file, format='mp3')
            os.remove(downloaded_file)
            print(f"Converted to MP3: {mp3_file}")
        else:
            print("Already in MP3 format.")

    except Exception as e:
        print(f"Error: {e}")


def clean_youtube_url(url):
    print(url)
    
    # Fix shortened or long URLs with parameters
    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    elif "watch?v=" in url:
        video_id = url.split("watch?v=")[1].split("&")[0]
    else:
        video_id = url  # fallback, unlikely

    print(f"https://www.youtube.com/watch?v={video_id}")
    
    return f"https://www.youtube.com/watch?v={video_id}"



if __name__ == "__main__":
    url = input("Enter YouTube URL: ")
    # fixed_url = clean_youtube_url(url.strip())
    download_audio_as_mp3(url)
