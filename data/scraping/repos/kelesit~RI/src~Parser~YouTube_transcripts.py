from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader


if __name__ == "__main__":

    # Two Karpathy lecture videos
    urls = ["https://youtu.be/kCc8FmEb1nY", "https://youtu.be/VMj-3S1tku0"]

    # Directory to save audio files
    save_dir = "~/Downloads/YouTube"

    # Transcribe the videos to text
    loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())
    docs = loader.load()
    print(docs[0].page_content[0:500])