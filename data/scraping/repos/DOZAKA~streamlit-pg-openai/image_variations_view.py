import streamlit
from . import ContentView
from streamlit_pg.modules.openai_manager.openai_manager import OpenAIManager


class ImageVariationsView(ContentView):
    def __init__(self, st: streamlit, oai_manager: OpenAIManager):
        super().__init__(st, oai_manager)
        self.openai_doc: str = "https://platform.openai.com/docs/guides/images/variations"

    def view(self) -> None:
        self.st.write("OpenAI API Introduction [link](%s)" % self.openai_doc)
        self.st.header("[Request]")
        uploaded_file = self.st.file_uploader(label='Pick an image to generate image')

        if not uploaded_file:
            return None

        image_data: any = uploaded_file.getvalue()

        self.st.image(image_data)
        is_clicked: bool = self.st.button('Send Request')

        if not is_clicked:
            return

        self.st.header("[Response]")
        img_url: str = self.oai_manager.image_to_image_url(image_data)

        if not img_url:
            return

        self.st.image(img_url)
