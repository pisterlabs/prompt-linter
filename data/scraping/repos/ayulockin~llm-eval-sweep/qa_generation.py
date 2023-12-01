import os
import wandb
import random
import argparse
import pandas as pd
from tqdm import tqdm

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import QAGenerationChain
from langchain.llms import Cohere

from dotenv import load_dotenv
load_dotenv("/Users/ayushthakur/integrations/llm-eval/apis.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

data_pdf = "data/qa/2304.12210.pdf"

def get_args():
    parser = argparse.ArgumentParser(description="Train image classification model.")
    parser.add_argument(
        "--pdf_path", type=str, default="data/qa/2304.12210.pdf", help="path to the PDF file."
    )
    parser.add_argument(
        "--num_questions", type=int, default=100, help="Number of questions to be generated."
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="llm-eval-sweep",
        help="wandb project name",
    )

    return parser.parse_args()

# Build question-answer pairs for evaluation
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, 
    chunk_overlap  = 100,
    length_function = len,
)

loader = PyPDFLoader(data_pdf)
qa_chunks = loader.load_and_split(text_splitter=text_splitter)
print("Number of chunks for building qa eval set:", len(qa_chunks))

templ = """You are a smart assistant designed to come up with meaninful question and answer pair. The question should be to the point and the answer should be as detailed as possible.
Given a piece of text, you must come up with a question and answer pair that can be used to evaluate a QA bot. Do not make up stuff. Stick to the text to come up with the question and answer pair.
When coming up with this question/answer pair, you must respond in the following format:
```
{{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}}
```

Everything between the ``` must be valid json.

Please come up with a question/answer pair, in the specified JSON format, for the following text:
----------------
{text}"""

PROMPT = PromptTemplate.from_template(templ)


def main(args: argparse.Namespace):
    qa_pairs = []

    # Initialize wandb
    wandb.init(project=args.wandb_project_name, config=vars(args))

    # Load OpenAI's `gpt-3.5-turbo`
    gpt_llm = ChatOpenAI(temperature=0.8)
    gpt_chain = QAGenerationChain.from_llm(llm=gpt_llm, prompt=PROMPT)

    # Load Cohere's `command` model
    command_llm = Cohere(model="command", temperature=0.8)
    command_chain = QAGenerationChain.from_llm(llm=command_llm, prompt=PROMPT)

    # Generate random ints that will be used to index the chunks.
    random_chunks = []
    for i in range(args.num_questions):
        random_chunks.append(random.randint(5, 172)) # To avoid title, content, and references pages.
    
    # Half questions will be generated using `gpt-3.5-turbo` and half using `command`.
    div_num = args.num_questions//4
    div_num = int(args.num_questions - div_num)

    # Generate QA using `gpt-3.5-turbo`
    for idx in tqdm(random_chunks[:div_num], desc="QA Generation using OpenAI..."):
        qa = gpt_chain.run(qa_chunks[idx].page_content)
        assert len(qa) == 1
        assert isinstance(qa[0], dict)
        qa[0]["model"] = "openai: gpt-3.5-turbo"
        qa_pairs.extend(qa)

    # Generate QA using `command`
    for idx in tqdm(random_chunks[div_num:], desc="QA Generation using Cohere..."):
        qa = command_chain.run(qa_chunks[idx].page_content)
        assert len(qa) == 1
        assert isinstance(qa[0], dict)
        qa[0]["model"] = "cohere: command"
        qa_pairs.extend(qa)

    qa_df = pd.DataFrame(qa_pairs)
    wandb.log({"QA Eval Pair": qa_df})


if __name__ == "__main__":
    args = get_args()
    print(args)

    main(args)