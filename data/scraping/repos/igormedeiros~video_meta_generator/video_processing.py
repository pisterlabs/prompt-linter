import os

import moviepy.editor as mp

from excel_processing import save_to_excel
from logger import log
from openai_processing import generate_video_metadata
from transcription_processing import transcribe_audio


def extract_audio_from_video(video_path, audio_path):
    """Extracts audio from a video and saves it as a WAV file."""
    log.debug(f'extract_audio_from_video({video_path}, {audio_path})')
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def process_videos(input_folder, output_file, video_file, language, progress_bar):
    log.debug(f'process_videos({input_folder}, {output_file}, '
              f'{video_file}, {language}, {progress_bar})')
    file_path = os.path.join(input_folder, video_file)
    audio_path = os.path.join(input_folder, f"{video_file}.wav")

    # Extract audio from video
    extract_audio_from_video(file_path, audio_path)

    # Transcribe audio
    transcription = transcribe_audio(audio_path, language)

    # Remove audio file
    os.remove(audio_path)

    if (transcription):
        # Generate metadata
        title, description, hashtags = generate_video_metadata(transcription)

        # Save metadata to spreadsheet
        save_to_excel(output_file, file_path, title,
                      description, hashtags)

    progress_bar.update(1)
