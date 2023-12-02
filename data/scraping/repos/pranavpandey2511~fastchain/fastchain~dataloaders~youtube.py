from langchain.document_loaders import YoutubeLoader
from base import BaseDataloader


class YoutubeDataLoader(BaseDataloader):
    """Yotube dataloader

    Args:
        YoutubeLoader (_type_): _description_
    """
    
    def __init__(self):
        
        loader = YoutubeLoader
        pass
        video_id: str,
        add_video_info: bool = False,
        language: Union[str, Sequence[str]] = "en",
        translation: str = "en",
        continue_on_failure: bool = False,