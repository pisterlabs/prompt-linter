from pathlib import Path
import youtube_dl
import librosa
import numpy as np
from pydub import AudioSegment
import openai
from config import *
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
import random


openai.api_key = API_KEY_OPENAI


class VideoBot:
    def __init__(
        self,
        song_url,
        bg_url,
        results_dir,
        downloads_dir,
        lyrics_dir,
        length=60,
        font_size=38,
        snippets=5,
        padding=10,
        bitrate="5000k",
        verbose=False,
    ):
        self.song_url = song_url
        self.bg_url = bg_url

        self.song_id = song_url.split("=")[-1]
        self.bg_id = bg_url.split("=")[-1]

        self.songs_dir = Path(downloads_dir / "songs")
        self.backgrounds_dir = Path(downloads_dir / "backgrounds")
        self.results_dir = Path(results_dir)
        self.downloads_dir = Path(downloads_dir)
        self.lyrics_dir = Path(lyrics_dir)

        self.song_dir = self.songs_dir / f"{self.song_id}.mp3"
        self.bg_dir = self.backgrounds_dir / f"{self.bg_id}.mp4"
        self.lyric_dir = self.lyrics_dir / f"{self.song_id}.srt"

        self.song_title = None
        self.bg_title = None
        self.artist = None
        self.title = None
        self.font_size = font_size
        self.length = length
        self.snippets = snippets
        self.padding = padding
        self.bitrate = bitrate
        self.verbose = verbose

        # verify if the directories are dirs and exist
        if not self.results_dir.exists():
            self.results_dir.mkdir()
        if not self.downloads_dir.exists():
            self.downloads_dir.mkdir()
            if not self.songs_dir.exists():
                self.songs_dir.mkdir()
            if not self.backgrounds_dir.exists():
                self.backgrounds_dir.mkdir()
        if not self.lyrics_dir.exists():
            self.lyrics_dir.mkdir()

    def download(self):
        def mp3():
            ydl_opts = {
                "outtmpl": str(self.songs_dir / f"{self.song_id}.%(ext)s"),
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
                "nocheckcertificate": True,
                "quiet": True,
                "progress": True,
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(self.song_url, download=False)
                self.title = info_dict["track"]
                self.artist = info_dict["artist"]
                print(f"-> Running LyricBot for {self.title} by {self.artist}...")
                if Path(self.song_dir).exists():
                    if self.verbose:
                        print("-> Song altready downloaded! Continuing...")
                else:
                    if self.verbose:
                        print("-> Downloading song...")
                    ydl.download([self.song_url])
                self.song_title = info_dict["title"]

        def mp4():
            ydl_opts = {
                "outtmpl": str(self.backgrounds_dir / f"{self.bg_id}.%(ext)s"),
                "format": "bestvideo[height=1080][ext=mp4]",
                "postprocessors": [
                    {
                        "key": "FFmpegVideoConvertor",
                        "preferedformat": "mp4",
                    }
                ],
                "nocheckcertificate": True,
                "quiet": True,
                "progress": True,
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(self.bg_url, download=False)
                if Path(self.bg_dir).exists():
                    if self.verbose:
                        print("-> Background altready downloaded! Continuing...")
                else:
                    if self.verbose:
                        print("-> Downloading background...")
                    ydl.download([self.bg_url])

                self.bg_title = info_dict["title"]

        mp3()
        mp4()

    def crop_audio(self):
        # Load the audio file
        audio_file = self.song_dir

        try:
            y, sr = librosa.load(audio_file)
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {e}")

        if librosa.get_duration(y=y) <= 60:
            print("-> Song is altready cropped! Skipping...")
            return

        # Compute the spectrogram of the audio signal
        S = np.abs(librosa.stft(y))
        band_means = np.mean(S, axis=1)
        max_index = np.argmax(band_means)
        start_time = round(librosa.frames_to_time(np.argmax(S[max_index, :])))
        end_time = round(start_time + self.length)

        audio_segment = AudioSegment.from_file(audio_file)
        extracted_segment = audio_segment[start_time * 1000 : end_time * 1000]
        outfile = self.song_dir
        extracted_segment.export(outfile, format="mp3")

        if self.verbose:
            print(f"-> Cropped song from {start_time:.2f} to {end_time:.2f} seconds.")

    def generate_lyrics(self):
        file = self.song_dir
        audio_file = open(file, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file, response_format="srt").strip()

        open(self.lyric_dir, "w").write(transcript)

        pieces = [line for line in transcript.split("\n") if line != ""]
        string = f"\n\n{int(pieces[-3])+1}\n{pieces[-2][-12:]} --> 00:02:00,000"

        open(self.lyric_dir, "a").write(string)

    def assemble_video(self):
        clip = VideoFileClip(str(self.bg_dir), verbose=False)
        valid_start_time = self.padding
        valid_end_time = clip.duration - self.padding - self.snippets

        # Define the list of snippet start times
        snippet_start_times = []
        while sum([self.snippets for start in snippet_start_times]) < self.length:
            # Generate a random start time within the valid range
            start_time = random.uniform(valid_start_time, valid_end_time)
            # Check if the start time is too close to any existing snippet
            if all(abs(start_time - s) >= self.snippets for s in snippet_start_times):
                snippet_start_times.append(start_time)
        # Create the list of snippet clips
        snippet_clips = []
        for start_time in snippet_start_times:
            end_time = start_time + self.snippets
            snippet_clip = clip.subclip(start_time, end_time)
            snippet_clips.append(snippet_clip)
        # Concatenate the snippet clips to form the final clip
        final_clip = concatenate_videoclips(snippet_clips)

        # Resize the final clip
        new_height = 1280
        new_width = int(final_clip.w * (new_height / final_clip.h))
        final_clip = final_clip.resize(width=new_width, height=new_height)

        # Create a new black background clip with the desired frame dimensions
        background = ColorClip(size=(720, 1280), color=(0, 0, 0))

        # Calculate the position of the top-left corner of the final clip in the new frame
        x_pos = (720 - final_clip.w) // 2
        y_pos = 0

        # Overlay the final clip on top of the black background clip at the desired position
        final_clip = CompositeVideoClip([background, final_clip.set_position((x_pos, y_pos))])

        # Create the subtitles clip
        generator = lambda txt: TextClip(
            txt,
            font="HelveticaNeueLTStd-HvCn",
            fontsize=self.font_size,
            color="white",
            stroke_color="black",
            stroke_width=1,
            method="caption",
            size=(700, 700),
        )
        subs = SubtitlesClip(str(self.lyric_dir), generator)
        subtitles = SubtitlesClip(subs, generator)

        # Overlay the subtitles on top of the final clip
        final_clip = CompositeVideoClip([final_clip, subtitles.set_pos(("center", "center"))])

        # Set the duration of the final clip
        final_clip = final_clip.set_duration(self.length)

        # Load the audio file
        audio_clip = AudioFileClip(str(self.song_dir))
        final_clip = final_clip.set_audio(audio_clip)

        if self.artist and self.title:
            output_file = str(self.results_dir / f"{self.artist} - {self.title}.mp4")
        else:
            output_file = str(self.results_dir / f"{self.song_title}.mp4")

        if self.verbose:
            print(f"-> Assembling Lyric Video...")

        final_clip.write_videofile(
            str(output_file),
            fps=clip.fps,
            threads=8,
            preset="ultrafast",
            codec="libx264",
            bitrate=self.bitrate,
            audio_codec="aac",
            logger="bar",
        )

    def run(self):
        bot.download()
        bot.crop_audio()
        bot.generate_lyrics()
        bot.assemble_video()


if __name__ == "__main__":
    bg_url = "https://www.youtube.com/watch?v=" + ""
    song_url = "https://www.youtube.com/watch?v=" + ""

    cwd = Path.cwd()
    results_dir = cwd / "results"
    downloads_dir = cwd / "downloads"
    lyrics_dir = cwd / "lyrics"

    bot = VideoBot(
        bg_url=bg_url,
        song_url=song_url,
        results_dir=results_dir,
        downloads_dir=downloads_dir,
        lyrics_dir=lyrics_dir,
        length=60,
        font_size=38,
        padding=50,
        snippets=5,
        bitrate="3000k",
        verbose=True,
    )

    bot.download()
    # bot.crop_audio()
    # bot.generate_lyrics()
    bot.assemble_video()
