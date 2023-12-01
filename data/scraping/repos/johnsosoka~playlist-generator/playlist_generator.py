from langchain import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
import faiss
from config.config_loader import ConfigLoader
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# configure
config_loader = ConfigLoader("config.yml")
config_loader.set_environment_variables()
config = config_loader.load_config()

from src.tools.add_song_tool import AddSongTool
from src.tools.playlist_content_tool import PlaylistContentsTool
from tools.find_song_tool import FindSongTool

tools = load_tools(["google-search"])
tools += [
    FindSongTool(),
    AddSongTool(),
    PlaylistContentsTool()
]

agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    memory=vectorstore.as_retriever(),
)
# Set verbose to be true
agent.chain.verbose = True

task_template = """
Your task is to build a themed spotify playlist. The playlist must not contain any duplicate songs. To add
a song to a spotify playlist, you must identify the URI.

1. Identify songs that fit the theme: '{song_theme}' using existing knowledge and internet search.
2. Find the URI for the song on spotify.
3. Check the playlist contents to ensure that the song is not already in the playlist. If the song is already in the playlist DO NOT ADD IT. Find another song.
4. Add the song to playlist id {playlist_id}

Your task is complete when the playlist has {num_items} songs in it.

Remember that it is essential that only unique songs are added to the playlist. Check the playlist contents before adding a song
to ensure that it is not already in the playlist. If the song is already in the playlist and you add it again, you will be penalized.
"""

prompt = PromptTemplate.from_template(task_template)


playlist_id = "0ylrX64UMWUwS1gjrDY2UO"
topic = "songs about mountains"
target_playlist_size = 5


agent.run([prompt.format(song_theme=topic, playlist_id=playlist_id, num_items=target_playlist_size)])

