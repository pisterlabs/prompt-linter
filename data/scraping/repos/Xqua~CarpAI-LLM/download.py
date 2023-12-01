from huggingface_hub import hf_hub_download
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import os

HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

# Replace this if you want to use a different model
model_id, filename = ("lmsys/fastchat-t5-3b-v1.0", "pytorch_model.bin") 

downloaded_model_path = hf_hub_download(
    repo_id=model_id,
    filename=filename,
    token=HUGGING_FACE_API_KEY,
)

print(downloaded_model_path)

model_id = "lmsys/fastchat-t5-3b-v1.0"
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text2text-generation",
    model_kwargs={"temperature": 0, "max_length": 1000},
)
