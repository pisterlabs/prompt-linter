from langchain.chains.summarize import load_summarize_chain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
# import openai
import time
import langchain
langchain.debug = True
load_dotenv()
# openai.api_key = os.environ.get("OPENAI_API_KEY")
n_gpu_layers = 200  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
n_ctx = 4096 


start_time = time.time()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
openai_chat = LlamaCpp(
    model_path="ggml-model-q4_1.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=n_ctx,
    callback_manager=callback_manager
)


def make_summarize(filename, model_params):
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()
    # Split the source text
    text_splitter = CharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=100,
        length_function=len,
        separator=""
    )
    docs = text_splitter.create_documents([data])
    # Callbacks support token-wise streaming
    prompt_template = """Act as a professional technical meeting minutes writer.
    Tone: formal
    Format: Technical meeting summary
    Tasks:
    - highlight action items and owners
    - highlight the agreements
    - Use bullet points if needed
    {text}
    ПИШИ ОТВЕТ НА РУССКОМ ЯЗЫКЕ:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    refine_template = (
        "Пиши ответ на русском языке\n"
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary in Russian: following the format"
        "Discussed: <Discussed-items>"
        "Follow-up actions: <a-list-of-follow-up-actions-with-owner-names>"
        "If the context isn't useful, return the original summary. Highlight agreements and follow-up actions and owners."
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    chain = load_summarize_chain(
        openai_chat,
        chain_type="map_reduce",
        return_intermediate_steps=True,
        map_prompt=PROMPT,
        combine_prompt=PROMPT,
    )

    resp = chain({"input_documents": docs}, return_only_outputs=True)

    return resp["output_text"]



