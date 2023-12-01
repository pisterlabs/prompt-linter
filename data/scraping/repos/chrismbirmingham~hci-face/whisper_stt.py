"""Module wraps the Whisper Model from OpenAI to transcribe audio

The whisper model provides state of the art results at a passable speed.
"""

import os
import tempfile
import csv
import argparse
import whisper
from pydub import AudioSegment


class WhisperSTT:
    """ Helper class which transcribes audio clips or files
    """

    def __init__(self, model_size: str = "medium", save_dir: str = None) -> None:
        self.audio_model = whisper.load_model(model_size)

        if not save_dir:
            self.save_dir = tempfile.mkdtemp()
        else:
            self.save_dir = save_dir

    def transcribe_clip(self, audio_clip: AudioSegment) -> str:
        """Transcribes audio segment

            Args:
                audio_clip (AudioSegment): bytes read from a file containing speech

            Returns:
                str: the transcribed text. """
        default_wave_path = os.path.join(self.save_dir, "temp.wav")
        audio_clip.export(default_wave_path, format="wav")
        result = self.audio_model.transcribe(default_wave_path, language='english')
        return result["text"]

    def transcribe_file(self, file_path: str, csv_name: str="transcription_test.csv") -> dict:
        """Transcribe a file"""
        result = self.audio_model.transcribe(file_path, language='english')

        transcription_path = os.path.join(self.save_dir, csv_name)

        self.save_csv(result["segments"], transcription_path)
        return result

    def save_csv(self, segments, filename="speech_segments.csv"):
        """Save csv of speech segments"""
        with open(filename, 'w') as my_file:
            writer = csv.writer(my_file)
            writer.writerow(segments[0].keys())
            for seg in segments:
                writer.writerow(seg.values())
            my_file.close()


def main():
    """Test of WhisperSTT"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model", default="base", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--filename", default="output/test.wav",
                        help="location of file to transcribe")

    args = parser.parse_args()

    transcriber = WhisperSTT(model_size=args.model, save_dir="output")
    print("loading complete")

    result = transcriber.transcribe_file(args.filename)
    print(result['text'])


if __name__ == "__main__":
    main()
