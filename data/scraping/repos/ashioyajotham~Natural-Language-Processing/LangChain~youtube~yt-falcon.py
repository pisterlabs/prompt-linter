from IPython.display import YouTubeVideo

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain
import torch
import langchain

# load video
loader = YoutubeLoader.from_youtube_url('https://www.youtube.com/watch?v=Y_O-x-itHaU')
text = loader.load()

# split text into sentences
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
sentences = splitter.split_documents(text)


# load language model
model_repo = 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2'
tokenizer = AutoTokenizer.from_pretrained(model_repo)
model = AutoModelForCausalLM.from_pretrained(model_repo,
                                                load_in_8bit=True, # This will load the model in 8-bit precision, which will make it faster and use less memory
                                                device_map='auto', # This will use the GPU if available, otherwise CPU
                                                torch_dtype=torch.float16, 
                                                low_cpu_mem_usage=True,
                                                trust_remote_code=True)

print(model.get_memory_footprint())

max_len = 2048
task = 'text-generation'
T = 0

# create pipeline
pipe = pipeline(
    task = task,
    model = model,
    tokenizer = tokenizer,
    max_length = max_len,
    temperature = T,
    top_p = .95,
    repetition_penalty = 1.2,
    pad_token_id = 11
)

llm = HuggingFacePipeline(pipeline=pipe)


chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=True)

# default prompt template
chain.llm_chain.prompt.template

summary = chain.run(text)


# custom prompt template
chain2 = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=False)

# change the prompt template
chain2.llm_chain.prompt.template = \
"""Write a three paragraph summary of the following text:
"{input_text}"
3 PARAGRAPH SUMMARY:"""

summary = chain2.run(text)

len(summary)