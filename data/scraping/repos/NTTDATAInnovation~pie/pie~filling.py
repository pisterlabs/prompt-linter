from pie.config import AZURE_OPENAI_CREDENTIALS
from langchain.llms.openai import AzureOpenAI

####################
# LLM
####################

LLM = AzureOpenAI(
    # model configuration
    verbose=True,
    temperature=0,  # For reproducibility
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    n=1,
    best_of=1,
    batch_size=1,  # TODO: A better strategy than per-document?"""
    request_timeout=None,  # Default is 600
    max_retries=6,
    streaming=False,
    # credentials
    **AZURE_OPENAI_CREDENTIALS
)
