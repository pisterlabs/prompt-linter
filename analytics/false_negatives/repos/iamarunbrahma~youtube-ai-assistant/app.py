import warnings

warnings.filterwarnings("ignore")
import os, requests, openai, cohere
import gradio as gr
from pathlib import Path
from langchain.document_loaders import YoutubeLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain

COHERE_API_KEY = os.environ["COHERE_API_KEY"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_CLUSTER_URL = os.environ["QDRANT_CLUSTER_URL"]
QDRANT_COLLECTION_NAME = os.environ["QDRANT_COLLECTION_NAME"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
prompt_file = "prompt_template.txt"


def yt_loader(yt_url):
    res = requests.get(f"https://www.youtube.com/oembed?url={yt_url}")
    if res.status_code != 200:
        yield "Invalid Youtube URL. Kindly, paste here a valid Youtube URL."
        return

    yield "Extracting transcript from youtube url..."
    loader = YoutubeLoader.from_youtube_url(yt_url, add_video_info=True)
    transcript = loader.load()

    video_id = transcript[0].metadata["source"]
    title = transcript[0].metadata["title"]
    author = transcript[0].metadata["author"]

    docs = []
    for i in range(len(transcript)):
        doc = Document(page_content=transcript[i].page_content)
        docs.append(doc)

    yield "Splitting transcript into chunks of text..."
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",
        chunk_size=1024,
        chunk_overlap=64,
        separators=["\n\n", "\n", " "],
    )

    docs_splitter = text_splitter.split_documents(docs)
    cohere_embeddings = CohereEmbeddings(model="large", cohere_api_key=COHERE_API_KEY)

    yield "Uploading chunks of text into Qdrant..."
    qdrant = Qdrant.from_documents(
        docs_splitter,
        cohere_embeddings,
        url=QDRANT_CLUSTER_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION_NAME,
    )

    with open(prompt_file, "r") as file:
        prompt_template = file.read()

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["question", "context"]
    )

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY
    )
    global qa
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=qdrant.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
    )

    yield "Generating summarized text from transcript..."
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
    summarized_text = chain.run(docs_splitter)
    res = (
        "Video ID: "
        + video_id
        + "\n"
        + "Video Title: "
        + title
        + "\n"
        + "Channel Name: "
        + author
        + "\n"
        + "Summarized Text: "
        + summarized_text
    )
    yield res


def chat(chat_history, query):
    res = qa.run(query)
    progressive_response = ""

    for ele in "".join(res):
        progressive_response += ele + ""
        yield chat_history + [(query, progressive_response)]


with gr.Blocks() as demo:
    gr.HTML("""<h1>Welcome to AI Youtube Assistant</h1>""")
    gr.Markdown(
        "Generate transcript from youtube url. Get a summarized text of the video transcript and also ask questions to AI Youtube Assistant.<br>"
        "Click on 'Build AI Bot' to extract transcript from youtube url and get a summarized text.<br>"
        "After summarized text is generated, click on 'AI Assistant' tab and ask queries to the AI Assistant regarding information in the youtube video."
    )

    with gr.Tab("Load/Summarize Youtube Video"):
        text_input = gr.Textbox(
            label="Paste a valid youtube url",
            placeholder="https://www.youtube.com/watch?v=AeJ9q45PfD0",
        )
        text_output = gr.Textbox(label="Summarized transcript of the youtube video")
        text_button = gr.Button(value="Build AI Bot!")
        text_button.click(yt_loader, text_input, text_output)

    with gr.Tab("AI Assistant"):
        chatbot = gr.Chatbot()
        query = gr.Textbox(
            label="Type your query here, then press 'enter' and scroll up for response"
        )
        chat_button = gr.Button(value="Submit Query!")
        clear = gr.Button(value="Clear Chat History!")
        clear.style(size="sm")
        query.submit(chat, [chatbot, query], chatbot)
        chat_button.click(chat, [chatbot, query], chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)


demo.queue().launch()
