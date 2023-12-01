from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import xformers
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools
from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import initialize_agent
from langchain.chains import RetrievalQA

# This was inserted by me
import torch
# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

from constants import CHROMA_SETTINGS

from torch import cuda, bfloat16
import transformers

def run_model(query):
    # lmsys/vicuna-13b-delta-v1.1  , bigscience/bloom-560m ,garage-bAInd/Platypus2-70B-instruct
    # meta-llama/Llama-2-70b-chat-hf
    model_id ='bigscience/bloom-560m'
    # model_id = ''
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # Set quantization configuration to load large model with less GPU memory  - Cannot use quantization in Windows
    # bnb_config = transformers.BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=bfloat16
    # )

    # Initialize model configuration and model
    hf_token = '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'
    model_config = transformers.AutoConfig.from_pretrained(model_id, use_auth_token=hf_token)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        # quantization_config=bnb_config,
        device_map='auto'
    )
    model.eval()

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    # Define text generation pipeline
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,  # Langchain expects the full text
        task='text-generation',
        temperature=1.0,  # 'Randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # Max number of tokens to generate in the output
        repetition_penalty=1.1  # Without this, output begins repeating
    )

    llm = HuggingFacePipeline(pipeline=generate_text)

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    print("DB work started")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    print("db work ended")
    retriever = db.as_retriever()

    rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=retriever, return_source_documents=False
    )

    # Generate text based on the input query
    try:
        # Generate text based on the input query
        answer = rag_pipeline(query)
        print(answer)
        # ans, docs = answer['result'], answer['source_documents']
        return answer
    except Exception as e:
        # Handle any exceptions that may occur during the execution of the agent
        # print(f"Error while running the agent: {e}")
        return {e}
