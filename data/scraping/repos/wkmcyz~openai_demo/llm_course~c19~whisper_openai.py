import time
from hashlib import md5
from pathlib import Path
from typing import List

import openai, os

from util.proxy import set_proxy

cache_dir = Path("./data/cache")
cache_dir.mkdir(parents=True, exist_ok=True)


def set_openai():
    openai.api_key = os.getenv("OPENAI_API_KEY")


def inner_md5(string: str):
    return md5(string.encode("utf-8")).hexdigest()


def speech_to_text_whisper_api(mp3_file_path: Path, prompt: str = ""):
    cache_file_path = cache_dir.joinpath(f"speech_to_text_whisper_api_{inner_md5(str(mp3_file_path))}")
    if cache_file_path.exists():
        # read file and return
        print(f"read from cache : {cache_file_path}")
        return cache_file_path.read_text()
        return
    audio_file = open(mp3_file_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, prompt=prompt)
    result = transcript['text']
    cache_file_path.write_text(result)
    return result


def speech_to_text_whisper_model(mp3_file_path: Path, prompt: str = ""):
    cache_file_path = cache_dir.joinpath(f"speech_to_text_whisper_model_{inner_md5(str(mp3_file_path))}")
    if cache_file_path.exists():
        # read file and return
        print(f"read from cache {cache_file_path}")
        return cache_file_path.read_text()
    import whisper
    model = whisper.load_model("large")
    result = model.transcribe(str(mp3_file_path), initial_prompt=prompt)
    print(result)
    result = result['text']
    cache_file_path.write_text(result)
    return result


def split_mp3(mp3_file_path: Path, by_time_s: int) -> List[Path]:
    from pydub import AudioSegment
    result = []
    podcast = AudioSegment.from_mp3(mp3_file_path)
    # PyDub handles time in milliseconds
    time_millis = by_time_s * 1000
    total_length = len(podcast)
    start = 0
    index = 0
    while start < total_length:
        end = start + time_millis
        if end < total_length:
            chunk = podcast[start:end]
        else:
            chunk = podcast[start:]
        file_path = mp3_file_path.parent.joinpath(mp3_file_path.name.replace(".mp3", f"_slice_{index}.mp3"))
        if not file_path.exists():
            print(f"exporting {file_path}")
            with open(f"./data/podcast_clip_{index}.mp3", "wb") as f:
                chunk.export(f, format="mp3")
        else:
            print(f"{file_path} already exists")
        result.append(file_path)
        start = end
        index += 1
    return result


def texts_to_summary(texts: List[str], content_file_name: str = "cache_content_by_api.txt") -> str:
    from langchain.chat_models import ChatOpenAI
    from langchain.text_splitter import SpacyTextSplitter
    from llama_index import GPTListIndex, LLMPredictor, ServiceContext, SimpleDirectoryReader
    from llama_index.node_parser import SimpleNodeParser

    cache_content_file_path = Path(f"./data/{content_file_name}")
    if not cache_content_file_path.exists():
        with open(cache_content_file_path, "w") as f:
            f.write("\n".join(texts))

    # define LLM
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=1024))

    text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size=2048)
    parser = SimpleNodeParser(text_splitter=text_splitter)
    documents = SimpleDirectoryReader(str(cache_content_file_path)).load_data()
    nodes = parser.get_nodes_from_documents(documents)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    list_index = GPTListIndex(nodes=nodes, service_context=service_context)
    query_engine = list_index.as_query_engine(response_mode="tree_summarize")
    response = query_engine.query("请你用中文总结一下我们的播客内容:")
    return response


if __name__ == '__main__':
    set_proxy()
    set_openai()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    prompt = "这是一段中文播客。是《剑来》小说的节选。"
    # api : 0.06$
    begin = time.time()
    text = speech_to_text_whisper_api(Path(f"{script_dir}/data/jianlai_1.mp3"), prompt=prompt)
    # text = speech_to_text_whisper_model(Path(f"{script_dir}/data/jianlai_1.mp3"), prompt=prompt)
    text = speech_to_text_whisper_model(Path(f"{script_dir}/data/podcast_clip.mp3"), prompt=prompt)
    print(f"speech_to_text consumed : {time.time() - begin}")
    print("text:")
    print(text)
    print()
    begin = time.time()
    # summary = texts_to_summary([])
    print(f"summary consumed : {time.time() - begin}")
    # print("summary:")
    # print(summary)
