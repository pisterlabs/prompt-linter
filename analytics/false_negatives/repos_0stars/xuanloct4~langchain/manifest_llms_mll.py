import environment
import os

# !pip install manifest-ml

from manifest import Manifest
from langchain.llms.manifest import ManifestWrapper
# !FLASK_PORT=6000 python3 -m manifest.api.app \
#     --model_type huggingface \
#     --model_name_or_path EleutherAI/gpt-j-6B \
#     --device 0
# manifest = Manifest(
#     client_name = "huggingface",
#     client_connection = "http://127.0.0.1:6000"
# )

manifest = Manifest(
    client_name = "cohere",
    client_connection = "TiUABu14jqIoFjoYuMPFG8Sf71THEXNzQgbJsPOV",
)

# os.environ["TOMA_URL"]="https://staging.together.xyz/api"
# manifest = Manifest(
#     client_name="toma",
#     engine="Together-gpt-JT-6B-v1",
#     max_tokens=150,
#     top_p=0.9,
#     top_k=40,
#     stop_sequences=["\n"],
# )

from manifest import Manifest
from langchain.llms.manifest import ManifestWrapper
from langchain import ConversationChain, LLMChain, PromptTemplate


template = """I am a classification model. It will try to classify your input.

Input: {human_input}
Output:"""

prompt = PromptTemplate(
    input_variables=["human_input"], 
    template=template
)

chatgpt_chain = LLMChain(
    llm=ManifestWrapper(client=manifest), 
    prompt=prompt, 
    verbose=True
)

# output = chatgpt_chain.predict(human_input="Classes are \"positive\" and \"negative\". For example given\nInput: I love this product!\nOutput: positive.\nI think this movie was one of the worst of the year. Script was boring!")
# print(output)


# output = chatgpt_chain.predict(human_input="So awesome! I wish I could have gone")
# print(output)



print(manifest.client_pool.get_current_client().get_model_params())
llm = ManifestWrapper(client=manifest, llm_kwargs={"temperature": 0.001, "max_tokens": 256})
# Map reduce example
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain


_prompt = """Write a concise summary of the following:


{text}


CONCISE SUMMARY:"""
prompt = PromptTemplate(template=_prompt, input_variables=["text"])

text_splitter = CharacterTextSplitter()

mp_chain = MapReduceChain.from_params(llm, prompt, text_splitter)
with open('./documents/state_of_the_union.txt') as f:
    state_of_the_union = f.read()
print(mp_chain.run(state_of_the_union))




# from langchain.model_laboratory import ModelLaboratory

# manifest1 = ManifestWrapper(
#     client=Manifest(
#         client_name="huggingface",
#         client_connection="http://127.0.0.1:5000"
#     ),
#     llm_kwargs={"temperature": 0.01}
# )
# manifest2 = ManifestWrapper(
#     client=Manifest(
#         client_name="huggingface",
#         client_connection="http://127.0.0.1:5001"
#     ),
#     llm_kwargs={"temperature": 0.01}
# )
# manifest3 = ManifestWrapper(
#     client=Manifest(
#         client_name="huggingface",
#         client_connection="http://127.0.0.1:5002"
#     ),
#     llm_kwargs={"temperature": 0.01}
# )
# llms = [manifest1, manifest2, manifest3]
# model_lab = ModelLaboratory(llms)
# model_lab.compare("What color is a flamingo?")





