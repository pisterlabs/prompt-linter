from typing import List, Dict, Any, Optional, Tuple
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import mediainfo
from openai import OpenAI
from pathlib import Path
import datetime
import subprocess
import scrapetube
import argparse
import pandas as pd
import re
import os


def extract_video_info(video_data):
    """
    Extract the video URL and cleaned title from the video data.

    Parameters:
    - video_data: The raw video data from YouTube.

    Returns:
    - dict: Dictionary with "url" and "title" keys.
    """
    base_video_url = "https://www.youtube.com/watch?v="
    video_id = video_data["videoId"]
    video_title = video_data["title"]["runs"][0]["text"]
    return {"url": f"{base_video_url}{video_id}", "title": video_title}


def get_video_urls(
    channel_or_playlist_id: str, pattern: str = None
) -> List[Dict[str, str]]:
    """
    Get video URLs and cleaned video titles either from a channel with a specified pattern or from a playlist.
    """
    if "youtube.com/playlist" in channel_or_playlist_id:
        playlist_id = channel_or_playlist_id.split("?list=")[-1].split("&")[0]
        videos = scrapetube.get_playlist(playlist_id)
    else:
        videos = scrapetube.get_channel(channel_or_playlist_id)

    if pattern:
        videos = [
            video
            for video in videos
            if re.search(pattern, video["title"]["runs"][0]["text"])
        ]

    return [extract_video_info(video) for video in videos]


def get_video_urls_from_channel(channel_id: str, pattern: str = None) -> List[str]:
    """
    Get video URLs from a channel with a specified pattern.
    """
    return [video["url"] for video in get_video_urls(channel_id, pattern)]


def get_video_urls_from_playlist(playlist_id: str) -> List[str]:
    """
    Get video URLs from a playlist.
    """
    return [video["url"] for video in get_video_urls(playlist_id)]


class Transcriber:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize a Transcriber object.

        Parameters:
        - api_key (str) : OpenAI API key.
        """
        if api_key is not None:
            self.client = OpenAI(api_key=api_key)

        self.supported_formats = [
            "wav",
            "mp3",
            "ogg",
            "flv",
            "m4a",
            "mp4",
            "wma",
            "aac",
        ]

    def convert_mp4_to_m4a(self, input_file: str, output_file: str) -> None:
        """
        Converts an MP4 file to M4A.

        Parameters:
        - input_file (str): Path to the input MP4 file.
        - output_file (str): Path to save the converted M4A file.
        """
        try:
            command = [
                "ffmpeg",
                "-i",
                input_file,
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                output_file,
            ]
            subprocess.run(command, check=True)
        except Exception as e:
            print(f"Error converting MP4 to M4A: {e}")
            return None

    def preprocess_audio(
        self,
        input_file: str,
        output_file: str,
        audio_format: str = "m4a",
        channels: int = 1,
        frame_rate: int = 16000,
        min_silence_len: int = 500,
        silence_thresh: int = -40,
        keep_silence: int = 200,
    ) -> None:
        """
        Preprocesses audio by splitting on silence and exports to WAV.

        Parameters:
        - input_file (str): Path to the input audio file.
        - output_file (str): Path to save the processed audio file.
        - audio_format (str): Format of the input audio file.
        - channels (int): Desired number of channels for the processed audio.
        - frame_rate (int): Desired frame rate for the processed audio.
        - min_silence_len (int): Minimum length of silence to split on.
        - silence_thresh (int): Threshold value for silence detection.
        - keep_silence (int): Amount of silence to retain around detected non-silent chunks.
        """
        if audio_format not in self.supported_formats:
            raise ValueError(
                f"Unsupported audio format: {audio_format}. Supported formats are: {', '.join(self.supported_formats)}"
            )
        try:
            audio = (
                AudioSegment.from_file(input_file, format=audio_format)
                .set_channels(channels)
                .set_frame_rate(frame_rate)
            )
            audio_segments = split_on_silence(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=keep_silence,
            )
            processed_audio = AudioSegment.empty()
            for segment in audio_segments:
                processed_audio += segment
            processed_audio.export(output_file, format="wav")
        except Exception as e:
            print(f"Error during audio preprocessing: {e}")
            return None

    def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Transcribes audio using the Whisper model.

        Parameters:
        - audio_file_path (str): Path to the audio file to transcribe.

        Returns:
        - Dict[str, Any]: Transcription results.
        """
        try:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", file=Path(audio_file_path), response_format="text"
            )
            return transcript
        except Exception as e:
            print(f"Error during transcription: {e}")
            return {}

    def download_youtube_media(
        self,
        youtube_url: str,
        output_path: str,
        media_type: str = "audio",
    ) -> Optional[str]:
        """
        Downloads audio or video from a YouTube video using yt-dlp.

        Parameters:
        - youtube_url (str): URL of the YouTube video.
        - output_path (str): Path to save the downloaded audio or video.
        - media_type (str): Type of media to download, can be 'audio' or 'video'.
        - custom_format (str, optional): Custom format selector for yt-dlp (default=None).

        Returns:
        - Optional[str]: Name of the downloaded file or None if unsuccessful.
        """
        # Define default format selectors for yt-dlp
        format_selector = {
            "audio": "bestaudio[ext=m4a]",
            "video": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        }

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Command setup for yt-dlp
        cmd = [
            "yt-dlp",
            "-f",
            format_selector[media_type],  # Format selector based on media type
            "-o",
            os.path.join(output_path, "%(title)s.%(ext)s"),  # Output template
            youtube_url,
        ]

        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            output = result.stdout.decode("utf-8").strip()
            if result.returncode == 0:
                # Assuming the output filename is in the last line of stdout (depends on yt-dlp version)
                file_path = output.split("\n")[-1]
                print("Downloaded file path:", file_path)  # Debugging line
                return file_path
            else:
                print("yt-dlp command was not successful.")  # Debugging line
                print("Return code:", result.returncode)  # Debugging line
                print("Output:", output)  # Debugging line
                return None
        except subprocess.CalledProcessError as e:
            print("An exception occurred while running yt-dlp.")  # Debugging line
            print("Error output:", e.output.decode("utf-8"))  # Debugging line
            print("Error code:", e.returncode)  # Debugging line
            return None

    def download_multiple_youtube_media(
        self,
        youtube_urls: List[str],
        output_path: str,
        media_type: str = "audio",
        custom_format: Optional[str] = None,
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Downloads audio or video from a list of YouTube videos using download_youtube_media.

        Parameters:
        - youtube_urls (List[str]): List of URLs of the YouTube videos.
        - output_path (str): Path to save the downloaded audio or video files.
        - media_type (str): Type of media to download, can be 'audio' or 'video'.

        Returns:
        - List[Tuple[str, Optional[str]]]: A list of tuples where the first element is the YouTube URL and
        the second element is the name of the downloaded file or None if unsuccessful.
        """
        downloaded_files = []

        for url in youtube_urls:
            try:
                filename = self.download_youtube_media(
                    url, output_path, media_type, custom_format
                )
                downloaded_files.append((url, filename))
            except ValueError as ve:
                print(f"Failed to download {url}: {ve}")
                downloaded_files.append((url, None))
            except Exception as e:
                print(f"An error occurred while downloading {url}: {e}")
                downloaded_files.append((url, None))

        return downloaded_files

    def download_videos_from_source(
        self,
        source_id: str,
        output_path: str,
        media_type: str = "audio",
        pattern: str = None,
        custom_format: Optional[str] = None,
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Download media from YouTube videos given a source, which can be a channel or playlist ID or URL.

        Parameters:
        - source_id (str): YouTube channel ID, playlist ID, or URL.
        - output_path (str): Path to save the downloaded media.
        - media_type (str): Type of media to download, 'audio' or 'video'.
        - pattern (str, optional): Pattern to match video titles (for channels).

        Returns:
        - List[Tuple[str, Optional[str]]]: List of tuples containing the video URL and the filename of the downloaded media.
        """
        if "youtube.com/playlist" in source_id:
            video_urls = get_video_urls_from_playlist(source_id)
        else:
            video_urls = get_video_urls_from_channel(source_id, pattern)

        return self.download_multiple_youtube_media(
            video_urls, output_path, media_type, custom_format
        )

    def transcribe(
        self,
        input_file: str,
        output_file: str,
        audio_format: str = "m4a",
        save_to_csv: bool = False,
        use_preprocessing: bool = False,
        convert_from_mp4: bool = False,
    ) -> pd.DataFrame:
        """
        Manages the transcription process and returns results as a DataFrame.

        Parameters:
        - input_file (str): Path to the input audio file.
        - output_file (str): Path to save the transcriptions.
        - audio_format (str): Format of the input audio file.
        - save_to_csv (bool): Whether to save the results to a CSV file.
        - use_preprocessing (bool): Whether to preprocess the audio before transcription.
        - convert_from_mp4 (bool): Whether to convert the input from MP4 to M4A before transcription.

        Returns:
        - pd.DataFrame: Transcription results.
        """
        if convert_from_mp4:
            m4a_file = os.path.splitext(input_file)[0] + ".m4a"
            self.convert_mp4_to_m4a(input_file, m4a_file)
            input_file = m4a_file

        t1 = datetime.datetime.now()
        audio_file_to_transcribe = input_file

        if use_preprocessing:
            self.preprocess_audio(input_file, output_file, audio_format=audio_format)
            audio_file_to_transcribe = output_file

        output = self.transcribe_audio(audio_file_to_transcribe)
        t2 = datetime.datetime.now()

        output_df = pd.DataFrame([output])
        output_df["Start_Time"] = t1
        output_df["End_Time"] = t2
        output_df["Duration"] = (t2 - t1).total_seconds()

        if save_to_csv:
            output_df.to_csv(output_file, index=False)

        return output_df

    def transcribe_batch(
        self,
        input_files: List[str],
        output_folder: str,
        audio_format: str = "m4a",
        save_to_csv: bool = False,
    ) -> pd.DataFrame:
        """
        Batch transcribes a list of audio files.

        Parameters:
        - input_files (List[str]): List of paths to audio files to transcribe.
        - output_folder (str): Path to save the transcriptions.
        - audio_format (str): Format of the input audio files.
        - save_to_csv (bool): Whether to save the results to a CSV file.

        Returns:
        - pd.DataFrame: Batch transcription results.
        """
        dfs = []

        for i, input_file in enumerate(input_files):
            base_name = os.path.basename(input_file)
            file_name_without_extension = os.path.splitext(base_name)[0]
            output_file = os.path.join(
                output_folder, f"{file_name_without_extension}_processed.wav"
            )
            print(f"Processing file {i+1}/{len(input_files)}: {input_file}")
            df = self.transcribe(input_file, output_file, audio_format=audio_format)
            dfs.append(df)

        result_df = pd.concat(dfs, ignore_index=True)
        if save_to_csv:
            result_csv_path = os.path.join(output_folder, "batch_transcriptions.csv")
            result_df.to_csv(result_csv_path, index=False)
            print(f"Saved batch transcriptions to: {result_csv_path}")
        return result_df

    def extract_audio_from_video(self, video_file: str, output_audio_file: str) -> None:
        """
        Extracts audio from the video and saves it as a new file.

        Parameters:
        - video_file (str): Path to the video file.
        - output_audio_file (str): Path to save the extracted audio.
        """
        video = AudioSegment.from_file(video_file, format=mediainfo(video_file))
        video.export(output_audio_file, format="wav")

    def transcribe_video(
        self,
        video_file: str,
        output_file: str,
        save_to_csv: bool = False,
        use_preprocessing: bool = False,
    ) -> pd.DataFrame:
        """
        Manages the transcription of video by first extracting audio and then transcribing it.

        Parameters:
        - video_file (str): Path to the video file to transcribe.
        - output_file (str): Path to save the transcriptions.
        - save_to_csv (bool): Whether to save the results to a CSV file.
        - use_preprocessing (bool): Whether to preprocess the audio before transcription.

        Returns:
        - pd.DataFrame: Transcription results.
        """
        audio_file_path = video_file + "_audio.wav"
        self.extract_audio_from_video(video_file, audio_file_path)
        return self.transcribe(
            audio_file_path,
            output_file,
            audio_format="wav",
            save_to_csv=save_to_csv,
            use_preprocessing=use_preprocessing,
        )


def main():
    parser = argparse.ArgumentParser(description="Download media from YouTube.")
    parser.add_argument(
        "youtube_url", type=str, help="The YouTube URL to download from."
    )
    parser.add_argument(
        "output_path", type=str, help="The directory to save the downloaded media."
    )
    parser.add_argument(
        "--media_type",
        type=str,
        default="audio",
        choices=["audio", "video"],
        help="The type of media to download, audio or video.",
    )

    args = parser.parse_args()

    # Initialize the Transcriber
    transcriber = Transcriber()

    # Run the download_youtube_media method
    try:
        file_path = transcriber.download_youtube_media(
            args.youtube_url, args.output_path, args.media_type
        )
        if file_path:
            print(f"Download successful. File saved to: {file_path}")
        else:
            print("Download failed.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

# (.venve) (base) mohameddiomande@Mohameds-Air-2 movement_mixer % python audio.py
# "https://www.youtube.com/watch?v=wm_B2-V-S_0&list=PLIxQjHO1yTm99RG32st06TnxZHMgCbT4H" "/Users/mohameddiomande/Desktop/storage/dj/realease"
# "https://www.youtube.com/watch?v=P2WZx5LqZLQ&list=PLAFB_cBaNnFWQQHICZONgNGVJDmEfGGWf" "/Users/mohameddiomande/Desktop/storage/dj/realease"

# create a pip freeze file
# pip freeze > requirements.txt
# source .venv/bin/activate
