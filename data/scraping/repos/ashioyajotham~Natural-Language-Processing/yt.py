from IPython.display import YouTubeVideo
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain
import langchain

# load video
loader = YoutubeLoader.from_youtube_url('https://www.youtube.com/watch?v=Y_O-x-itHaU')
text = loader.load()

# split text into sentences
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
sentences = splitter.split_documents(text)

# Quantize the model
config = BitsAndBytesConfig.from_pretrained('h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2')
config.quantize_model = True

# create language model
from transformers import AutoModel
llm = HuggingFacePipeline(pipeline=None)
llm.config = config

# create chain
chain = LLMChain(llm=llm, chain_type="map_reduce", verbose=True)

# load language model
model_repo = 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2'
tokenizer = AutoTokenizer.from_pretrained(model_repo)
model = AutoModelForCausalLM.from_pretrained(model_repo,
                                                load_in_8bit=True,
                                                device_map='auto', # This will use the GPU if available, otherwise CPU
                                                torch_dtype=torch.int8,
                                                low_cpu_mem_usage=True,
                                                trust_remote_code=True)

max_len = 2048
task = 'text-generation'

# create pipeline
pipe = pipeline(
    task = task,
    model = model,
    tokenizer = tokenizer,
    max_length = max_len,
    temperature = 0,
    top_p = .95,
    repetition_penalty = 1.2,
    pad_token_id = 11
)

# generate text
generated_text = pipe(sentences[0])
print(generated_text[0]['generated_text'])