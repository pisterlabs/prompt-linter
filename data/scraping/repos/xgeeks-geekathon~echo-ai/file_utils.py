
from langchain.document_loaders import JSONLoader
import moviepy.editor as mp
import tempfile

def load_file(path):
    loader = JSONLoader(
        file_path=path,
        jq_schema='.text')

    data = loader.load()
    return data

def save_to_file(text, path):
    with open(path, 'w') as f:
        f.write(str(text))

def transform_video(file):
    clip = mp.VideoFileClip(file.name)
    temp = tempfile.mkstemp(prefix="audio_file", suffix=".mp3")

    clip.audio.write_audiofile(temp[1])
    
    with open(temp[1], 'r') as f:
        return f