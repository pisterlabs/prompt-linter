import os
import glob
text_paths = glob.glob('data/**/*.txt')
print(text_paths)

from dataclasses import dataclass
from typing import List


@dataclass
class Video:
    video_name: str
    text: str
    embedding: List[float]

from langchain.text_splitter import RecursiveCharacterTextSplitter
def split_text(text):
    separators = [".", "。", " "]
    splitter = RecursiveCharacterTextSplitter(separators=separators, chunk_size=2048, chunk_overlap=0)
    return splitter.split_text(text)


from dotenv import load_dotenv
load_dotenv()
import openai
from openai.embeddings_utils import get_embedding
import pickle
openai.api_key = os.getenv("OPENAI_API_KEY")
videos = []
try:
    for text_path in text_paths:
        with open(text_path, 'r') as f:
            text = f.read()
        # 'data/Lecture_19_-_Reward_Model_Linear_Dynamical_System_Stanford_CS229_-_Machine_Learning_Autumn_2018/text.txt'
        video_name = text_path.split('/')[1]
        text_splits = split_text(text)
        for text_split in text_splits:
            print(video_name)
            print(text_split)
            embedding = get_embedding(
                text_split,
                engine="text-embedding-ada-002"
            )
            video = Video(video_name=video_name, text=text_split, embedding=embedding)
            videos.append(video)
except Exception as e:
    with open(f'lecture_videos.pickle', 'wb') as f:
        pickle.dump(videos, f)

# ファイルにリストオブジェクトを保存する
with open(f'lecture_videos.pickle', 'wb') as f:
    pickle.dump(videos, f)