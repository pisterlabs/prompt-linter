from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import logging
import tiktoken
import time
import os

openai_api_key = os.environ.get("OPENAI_API_KEY")
# Summarization parameters
OPENAI_MODEL_NAME: str = "text-curie-001"
OPENAI_TOKENIZER_NAME: str = "cl100k_base"
OPENAI_MODEL_TEMPERATURE: float = 0.7  # 0 is fully deterministic, 1 is most random
OPENAI_MODEL_MAX_TOKENS: int = 500  # langchain automatically sets to max for OPENAI_MODEL_NAME
CHUNK_TOKEN_SIZE: int = 800
CHUNK_TOKEN_OVERLAP: int = 100

HALLUCINATION_PROMPT_TEMPLATE: str = (
    "You are a world class journalist."
    "Your job is to answer the following question."
    "'{summary}'"
    "The context for this follow up question is provided in this summary of your previous speech."
    "`{question}`"
    "Do not include irrelevant information."
    "Format your answer in complete sentences. Do not include fragments."
)
# Summarization helpers
encoder = tiktoken.get_encoding(OPENAI_TOKENIZER_NAME)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_TOKEN_SIZE, chunk_overlap=CHUNK_TOKEN_OVERLAP, length_function=lambda x: len(encoder.encode(x))
)
hallucination_prompt = PromptTemplate(template=HALLUCINATION_PROMPT_TEMPLATE, input_variables=["summary", "question"])

def create_hallucination(summary: str, question:str) -> str:
    """Returns the answer

    Arguments:
        summary {str} -- Article data dictionary. Needs keys url and content.
        openai_api_key {str} -- API key for the OpenAI API.
 """
    # transform the summary into a document form used by langchain
    # langchain_doc = Document(page_content=summary)

    # Initialize OpenAI LLM with langchain
    llm = OpenAI(
        model_name=OPENAI_MODEL_NAME,
        temperature=OPENAI_MODEL_TEMPERATURE,
        max_tokens=OPENAI_MODEL_MAX_TOKENS,
        openai_api_key=openai_api_key,
        batch_size=1
    )
    
    answer_chain = LLMChain(llm=llm, prompt=hallucination_prompt)
    answer = answer_chain.run({'summary':summary, 'question':question})
    return answer

def main():
    summary = "• European markets mixed as economic concerns dominate WEF in Davos \
            • Chinese economic data released overnight, GDP grew 3% in 2022\
            • Stoxx 600 flat, autos up 0.5%, retail down 0.5%\
            • Leaders of Spain, Latvia, Lithuania and Poland, CEOs of Unilever, UBS, Allianz, Swiss Re to speak at WEF\
            • Nasdaq Composite up 5.9%, S&P 500 up 4.2%, Dow up 3.5%\
            • China's December retail sales beat estimates, industrial output up 1.3%, economy expanded 2.9% in Q4"
    question = "What happened to European Markets?"
    answer = create_hallucination(summary, question)
    print(answer)

if __name__ == "__main__":
    main()