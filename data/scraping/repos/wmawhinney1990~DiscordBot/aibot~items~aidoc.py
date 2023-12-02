from pathlib import Path

from discord import app_commands ,ui, ButtonStyle, Interaction, SelectOption, Embed

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

from aibot.items.base_item import BaseItem
from aibot.config import Config
from aibot.utils import ContentTypes

class VectorStoreEngine:

    def __init__(self, settings, relative_path: Path = None):
        availible_embeddings = { "OpenAIEmbeddings": OpenAIEmbeddings}
        self._pointer = None
        self.embbedings = availible_embeddings[settings.VECTORSTORE_EMBEDDINGS]()
        self.chunk_size = settings.VECTORSTORE_CHUNK_SIZE
        self.chunk_overlap = settings.VECTORSTORE_CHUNK_OVERLAP
        if relative_path:
            self._embeddings_directory = relative_path / settings.EMBEDDINGS_DIRECTORY
            self._source_directory = relative_path / settings.SOURCEFILES_DIRECTORY
        else:
            self._embeddings_directory = settings.EMBEDDINGS_DIRECTORY
            self._source_directory = settings.SOURCEFILES_DIRECTORY

    def __repr__(self):
        return f"<VectorStorePointer pointer={self._pointer} chunk_size={self.chunk_size} chunk_overlap={self.chunk_overlap}>"
    def __str__(self):
        return str(self._pointer)

    @property
    def is_ready(self):
        return self._pointer is not None
    @property
    def pointer(self):
        return self._pointer
    @pointer.setter
    def pointer(self, embeddings_directory):
        if embeddings_directory.is_dir():
            self._pointer = embeddings_directory
        else:
            self._pointer = None
    @property
    def embedding_function(self):
        return self.embbedings

    @property
    def embeddings_directory(self):
        if not self._embeddings_directory.is_dir():
            self._embeddings_directory.mkdir(parents=False)
        return self._embeddings_directory
    @property
    def source_directory(self):
        if not self._source_directory.is_dir():
            self._source_directory.mkdir(parents=False)
        return self._source_directory
    @property
    def embeddings(self):     # Return a list of all the embeddings aviable
        return [ file for file in self.embeddings_directory.iterdir() if file.is_dir() ]
    @property
    def sources(self):        # Returns a list of source files in `source_directory`
        return [ file for file in self.source_directory.iterdir() if file.is_file() ]
    @property
    def unembedded_sources(self):     # Return a list of self.sources - self.embeddings
        return [ file for file in self.sources if file.stem not in list(map(lambda x: x.name, self.embeddings)) ]

    def create_embeddings(self, source_document):
        loader = PyPDFLoader(source_document)
        documents = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap).split_documents(loader.load())
        vectordb = Chroma.from_documents(documents, embedding=self.embedding_function, persist_directory=str(self.embeddings_directory / Path(source_document).stem))
        self.vdb = vectordb

    def query(self, query, chain_type=("map_reduce", "stuff")[0]):
        self.vdb = Chroma(persist_directory=str(self.pointer), embedding_function=self.embedding_function)
        chain = load_qa_chain(OpenAI(temperature=0), chain_type=chain_type)
        result= chain.run(input_documents=self.vdb.similarity_search(query), question=query)
        return result

    def set_pointer(self, document):
        # Error handle the pointer here
        # if the document is a pdf in the source folder and no copy in the enbeds folder, embed it
        # set_pointer should also set self.vdb, the Chroma Vector Stores
        embedding_of_interest = Path(document)

        if embedding_of_interest.parent != self.embeddings_directory:
            self._pointer == None

        if '/' in document:
            self.create_embeddings(document)

        self.pointer = self.embeddings_directory / Path(document).stem

class AIDoc(BaseItem):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.description: str = "Document loader for chatting with documents"
        self.config = Config.from_yaml(self.path / "config.yaml")
        self.vse = VectorStoreEngine(self.config, relative_path=self.path)

    @property
    def command(self):
        com = app_commands.commands.Command(name="aidoc", description="Upload / Query the AI docuement loader", callback=self.interact)
        app_commands.commands._populate_descriptions(com._params, { "prompt": f"Optional query prompt for [{self.vse.pointer}]" } ) 
        return com

    @property
    def discord_options(self):
        source_options = [ SelectOption(label=source_path.name, value=str(source_path), description="Embed this file.") for source_path in self.vse.unembedded_sources ]
        embed_options = [ SelectOption(label=vector_store.name, description="Vector Store to Query") for vector_store in self.vse.embeddings ]
        return embed_options + source_options

    @property
    def discord_embed(self):
        e = Embed(title="AI Document", description="User Interface for the AI Document Load and Query system. Dropdown will Embed and reference a file, or allow selection for reference.")
        selection = self.vse.pointer if self.vse.pointer is None else self.vse.pointer.stem
        e.add_field(name=f"Selected Vector DB",  value=selection, inline=True)
        return e

    @property
    def discord_view(self):
        async def dropdown_callback(document, interaction):
            self.vse.set_pointer(document)
            await interaction.response.edit_message(embed=self.discord_embed, view=self.discord_view)

        doc_dropper = DocumentDropdown(self.discord_options, dropdown_callback)
        return AIDocView(doc_dropper)           # Options: Embed from URL, Create Knowledge Cluster, Disban Knowledge Cluster

    async def save_attachement(self, attachment):
        filepath = self.vse.source_directory / attachment.filename
        await attachment.save(filepath)

    async def on_message(self, message, *args, **kwargs):
        if message.attachments:
            for attachment in message.attachments:
                if attachment.content_type == ContentTypes.PDF.value:
                    await self.save_attachement(attachment)
                    await message.reply(f"`{attachment.filename}` saved under AI Document Loader!")

    async def interact(self, interaction, prompt: str = None):
        self.i = interaction
        if prompt is not None:
            if self.vse.is_ready:
                await interaction.response.send_message(content=f"Querying [{self.vse.pointer.stem}] ...", ephemeral=True)
                result = self.vse.query(prompt)
                await interaction.followup.send(content=f"Input query[{self.vse.pointer.stem}]: {prompt}\n\nResult: {result}")
            else:
                await interaction.response.send_message(content="No Vector Store selected to query! Use the `/aidoc` command to see deatils or upload a PDF.")
        else:
            await interaction.response.send_message(content="AI Document Loader Interaction", embed=self.discord_embed, view=self.discord_view)

class DocumentDropdown(ui.Select):
    def __init__(self,options, dropdown_callback):
        super().__init__(placeholder='Document loader selection', options=options)
        # should check to see if callback is a Callable before setting
        self.outsourced_callback = dropdown_callback

    async def callback(self, interaction: Interaction):
        # The callback method is executed when a selection on the dropdown is made
        await self.outsourced_callback(self.values[0], interaction)

class AIDocView(ui.View):
    def __init__(self, *ui_items):
        super().__init__()
        for ui_element in ui_items:
            self.add_item(ui_element)