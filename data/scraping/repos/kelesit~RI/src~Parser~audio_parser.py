from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import audio


file = "/Users/huangsiyu/WORK/SteamAI/RID/src/data/jfk.flac"


def audio_parser(audio_path):
    loader = GenericLoader.from_filesystem(
        audio_path,
        parser=audio.OpenAIWhisperParser("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")
    )
    docs = loader.load()
    for doc in docs:
        print(doc.page_content)
    return docs


if __name__ == "__main__":
    result = audio_parser(file)



