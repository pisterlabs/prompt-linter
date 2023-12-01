# This file sets up the API keys and default language models for the examples.
# It includes both OpenAI and HuggingFace Language Models.

from dotenv import load_dotenv
import os

from langchain.llms import OpenAI, HuggingFaceHub

from utils.custom_stream import CustomStreamCallback

load_dotenv()

# Paid OpenAI API (https://platform.openai.com/account/api-keys)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Free HuggingFace API (https://huggingface.co/settings/tokens)
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

#region Instantiate Language Models
# - Different models can be used for different results/use cases
# - Temperature - 0-1: "randomness/diversity" of output (higher = more random)

# Paid OpenAI model (https://openai.com/blog/openai-api)
default_llm_open_ai = OpenAI(
    temperature=0.9,
    max_tokens=1000,
    streaming=True,
    callbacks=[CustomStreamCallback()] # Sets up output stream with colors
)

# Free HuggingFace model (https://huggingface.co/google/flan-t5-xl)
default_llm_hugging_face = HuggingFaceHub(
    repo_id="google/flan-t5-xl", 
    model_kwargs={
        "temperature": 0.6,
        "max_length": 64
    },
    callbacks=[CustomStreamCallback()] # Sets up output stream with colors
) 

# Sets default llm to OpenAI
default_llm = default_llm_open_ai

#endregion