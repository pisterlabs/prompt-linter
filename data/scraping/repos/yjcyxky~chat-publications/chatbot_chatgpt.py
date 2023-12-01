import gradio as gr
import click
import json
import os
from llama_index import SimpleDirectoryReader, ServiceContext, GPTVectorStoreIndex, LLMPredictor, PromptHelper, OpenAIEmbedding
from llama_index import StorageContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI

model_name = "text-davinci-003"

def check_variable():
    if os.environ.get("OPENAI_API_KEY") is None:
        print("Please set the OPENAI_API_KEY environment variable.")
        exit(1)


def launch_chatbot(persist_dir):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    # load index
    index = load_index_from_storage(storage_context).as_query_engine()

    def chatbot(input_text):
        print("Input: %s" % input_text)
        response = index.query(input_text)
        print("Response: %s" % response)
        return response.response.strip()

    return chatbot


@click.group()
def chatbot():
    pass


@chatbot.command(help="Build index from directory of documents.")
@click.option('--directory-path', '-d', required=True, help="The directory which saved the documents.")
def index(directory_path):
    check_variable()

    max_input_size = 2048
    num_outputs = 256
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(
        max_input_size, num_outputs, max_chunk_overlap,
        chunk_size_limit=chunk_size_limit
    )

    llm = ChatOpenAI(temperature=0.7, model_name=model_name,
                     max_tokens=num_outputs)
    llm_predictor = LLMPredictor(llm=llm)
    embedding = OpenAIEmbedding()

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        embed_model=embedding,
        prompt_helper=prompt_helper
    )
    doc_index = GPTVectorStoreIndex.from_documents(
        documents,
        service_context=service_context
    )
    doc_index = GPTVectorStoreIndex.from_documents(documents)

    doc_index.storage_context.persist(persist_dir=directory_path)
    metadata = {
        "max_input_size": max_input_size,
        "num_outputs": num_outputs,
        "max_chunk_overlap": max_chunk_overlap,
        "chunk_size_limit": chunk_size_limit,
        "directory_path": directory_path,
        "index_type": "GPTVectorStoreIndex",
        "model_name": "text-davinci-003",
        "temperature": 0.7,
        "max_tokens": num_outputs,
        "num_documents": len(documents),
        "document_names": [os.path.basename(file) for file in os.listdir(directory_path)]
    }

    filename = os.path.basename(directory_path)
    dirname = os.path.dirname(directory_path)
    metadata_filepath = os.path.join(dirname, f'{filename}_metadata.json')
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f)

    return index


@chatbot.command(help="Query index.")
@click.option('--directory-path', '-d', required=True, help="The directory which saved the documents.")
def query(directory_path):
    check_variable()

    if os.path.exists(directory_path):
        iface = gr.Interface(fn=launch_chatbot(directory_path),
                             inputs=gr.inputs.Textbox(lines=7,
                                                      label="Enter your text"),
                             outputs="text",
                             title="Custom-trained AI Chatbot")

        iface.launch(share=False)
    else:
        print("Index file not found.")
        return


if __name__ == "__main__":
    chatbot()
