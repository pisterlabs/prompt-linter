import marqo
from langchain.document_loaders import ImageCaptionLoader
import config


class ImagePocessor:
    def __init__(self, fileUrl):
        self.fileUrl = fileUrl
        self.client = marqo.Client(config.marqo_url)
        self.index_name = config.index_name

    def process(self):
        loader = ImageCaptionLoader([self.fileUrl])
        image = loader.load()
        caption = image[0].page_content.replace('[SEP]', '')
        img = self.client.index(self.index_name).add_documents([
            {
                "caption": caption,
                "image": self.fileUrl,
            }
        ], tensor_fields=['caption', 'image'])

        return img
