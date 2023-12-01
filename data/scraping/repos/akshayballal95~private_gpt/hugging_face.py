from langchain import ConversationChain, LLMChain, PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import  VectorstoreIndexCreator
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.llms import GPT4All
from langchain.llms import OpenAI
import os
from langchain.memory import VectorStoreRetrieverMemory
from langchain.callbacks import get_openai_callback

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import torch
import transformers


load_dotenv()

video_links = ["9lVj_DZm36c", "ZUN3AFNiEgc", "8KtDLu4a-EM"]

if os.path.exists('transcripts'):
    print('Directory already exists')
else:
    os.mkdir('transcripts')
for video_id in video_links:
    dir = os.path.join('transcripts', video_id)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    with open(dir+'.txt', 'w') as f:
     for line in transcript:
            f.write(f"{line['text']}\n")


loader = DirectoryLoader(path='./', glob = "**/*.txt", loader_cls=TextLoader,
                        show_progress=True)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
name = 'mosaicml/mpt-7b-instruct'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'torch'
config.init_device = 'cuda:0' # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True
)
from torch import cuda
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

# mtp-7b is trained to add "<|endoftext|>" at the end of generations
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    device=device,
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model will ramble
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=1024,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

from langchain.llms import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=generate_text)


index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])
retriever = index.vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)

# llm = GPT4All(model="./ggml-mpt-7b-instruct.bin", top_p=0.15, temp=0.3, repeat_penalty=1.1, n_ctx=4096)
# llm = OpenAI(temperature=0.5)

_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is smart and provides relevant information. If the AI does not know the answer to a question, it truthfully says it does not know.
Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""


PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)
conversation_with_summary = ConversationChain(
    llm=llm, 
    prompt=PROMPT,
    # We set a very low max_token_limit for the purposes of testing.
    # memory=memory,
)


while True:
    user_input = input("You: ")
    if user_input == "quit":
        break
    output=conversation_with_summary.predict(input = user_input)
    print(output)


# st.set_page_config(
#     page_title="YouTubeGPT",
#     page_icon=":robot:"
# )

# st.header("YouTubeGPT")
# st.markdown("[Github](https://github.com/akshayballal95)")

# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []

# if 'past' not in st.session_state:
#     st.session_state['past'] = []

# def get_text():
#     input_text = st.text_input("You: ","Hello, how are you?", key="input")
#     return input_text 

# user_input = get_text()

# if user_input:

#     output=conversation_with_summary.predict(input = user_input)
    
#     st.session_state.past.append(user_input)
#     st.session_state.generated.append(output)

# if st.session_state['generated']:

#     for i in range(len(st.session_state['generated'])-1, -1, -1):
#         message(st.session_state["generated"][i], key=str(i))
#         message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
