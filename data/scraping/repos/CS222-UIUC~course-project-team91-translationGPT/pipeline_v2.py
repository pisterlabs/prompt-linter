import argparse
import os
from tqdm import tqdm
import openai
from pytube import YouTube
import whisper

def arg_parse():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--link", help="youtube video link here", default=None, type=str, required=False)
    parser.add_argument("--video_file", help="local video path here", default=None, type=str, required=False)
    parser.add_argument("--audio_file", help="local audio path here", default=None, type=str, required=False)
    parser.add_argument("--srt_file", help="srt file input path here", default=None, type=str, required=False)  # New argument
    parser.add_argument("--download", help="download path", default='./downloads', type=str, required=False)
    parser.add_argument("--output_dir", help="translate result path", default='./results', type=str, required=False)
    parser.add_argument("--video_name", help="video name, if use video link as input, the name will auto-filled by youtube video name", default='placeholder', type=str, required=False)
    parser.add_argument("--model_name", help="model name only support text-davinci-003 and gpt-3.5-turbo", type=str, required=False, default="gpt-3.5-turbo")
    parser.add_argument("-only_srt", help="set script output to only .srt file", action='store_true')
    parser.add_argument("-v", help="auto encode script with video", action='store_true')
    
    args = parser.parse_args()
    return args

def youtube_download(download_path,link,video_file,audio_file):
    if link is not None and video_file is None:
    # Download audio from YouTube
        video_link = link
        video = None
        audio = None
        try:
            yt = YouTube(video_link)
            video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if video:
                video.download(f'{download_path}/video')
                print('Video download completed!')
            else:
                print("Error: Video stream not found")
            audio = yt.streams.filter(only_audio=True, file_extension='mp4').first()
            if audio:
                audio.download(f'{download_path}/audio')
                print('Audio download completed!')
            else:
                print("Error: Audio stream not found")
        except Exception as e:
            print("Connection Error")
            print(e) 
            exit()
    
    video_path = f'{download_path}/video/{video.default_filename}'
    audio_path = '{}/audio/{}'.format(download_path, audio.default_filename)
    audio_file = open(audio_path, "rb")
    if VIDEO_NAME == 'placeholder':
        VIDEO_NAME = audio.default_filename.split('.')[0]

    elif video_file is not None:
        # Read from local
        video_path = video_file
        if audio_file is not None:
            audio_file= open(audio_file, "rb")
            audio_path = audio_file
        else:
            os.system(f'ffmpeg -i {video_file} -f mp3 -ab 192000 -vn {download_path}/audio/{VIDEO_NAME}.mp3')
            audio_file= open(f'{download_path}/audio/{VIDEO_NAME}.mp3', "rb")
            audio_path = f'{download_path}/audio/{VIDEO_NAME}.mp3'

    return video_path, audio_path





def main(args):
    if args.link is None and args.video_file is None and args.srt_file is None:
        print("need video source or srt file")
        exit()

    # set up
    openai.api_key = os.getenv("OPENAI_API_KEY")
    DOWNLOAD_PATH = args.download
    if not os.path.exists(DOWNLOAD_PATH):
        os.mkdir(DOWNLOAD_PATH)
        os.mkdir(f'{DOWNLOAD_PATH}/audio')
        os.mkdir(f'{DOWNLOAD_PATH}/video')

    RESULT_PATH = args.output_dir
    if not os.path.exists(RESULT_PATH):
        os.mkdir(RESULT_PATH)

    VIDEO_NAME = args.video_name
    model_name = args.model_name




