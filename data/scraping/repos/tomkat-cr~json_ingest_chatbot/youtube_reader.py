from langchain.document_loaders import YoutubeLoader


def get_youtube_transcription(youtube_url, languages=['en', 'es']):
    loader = YoutubeLoader.from_youtube_url(
        youtube_url,
        add_video_info=True,
        language=languages
    )
    return loader.load()
