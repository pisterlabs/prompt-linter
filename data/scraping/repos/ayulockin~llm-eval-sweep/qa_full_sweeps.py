import os
import re
import json
import wandb
import numexpr
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from pydantic import BaseModel, Field, validator

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import QAGenerationChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import CohereEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import TFIDFRetriever
from langchain.vectorstores import FAISS
from langchain.llms import Cohere
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain
from evaluate import load

from dotenv import load_dotenv
load_dotenv("/Users/ayushthakur/integrations/llm-eval/apis.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")


def get_args():
    parser = argparse.ArgumentParser(description="Train image classification model.")
    parser.add_argument(
        "--embedding", type=str, help="name of the embedding class"
    )
    parser.add_argument(
        "--retriever", type=str, help="name of the vectorstore/retriever"
    )
    parser.add_argument(
        "--llm", type=str, help="name of the LLM model"
    )
    parser.add_argument(
        "--prompt_template_file", type=str, help="prompt template for the LLM"
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="llm-eval-sweep",
        help="wandb project name",
    )

    return parser.parse_args()


# Load the QA Eval dataset
wandb_api = wandb.Api()
run = wandb_api.run("ayush-thakur/llm-eval-sweep/2nrl2xh6")
artifact = run.use_artifact(wandb_api.artifact(name="ayush-thakur/llm-eval-sweep/run-2nrl2xh6-QAEvalPair:v0"))
download_dir = artifact.download()

with open(f"{download_dir}/QA Eval Pair.table.json") as f:
    data = json.load(f)
columns = data["columns"]
data = data["data"]

eval_df = pd.DataFrame(columns=columns, data=data)
eval_df = eval_df.sample(frac=1).reset_index(drop=True)

# Build question-answer pairs for evaluation
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, 
    chunk_overlap  = 100,
    length_function = len,
)

data_pdf = "data/qa/2304.12210.pdf"
loader = PyPDFLoader(data_pdf)
qa_chunks = loader.load_and_split(text_splitter=text_splitter)


def main(args: argparse.Namespace):
    # Initialize wandb
    wandb.init(project=args.wandb_project_name, config=vars(args))

    # Embedding
    if args.embedding == "SentenceTransformerEmbeddings":
        embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    elif args.embedding == "OpenAIEmbeddings":
        embedding = OpenAIEmbeddings()
    elif args.embedding == "CohereEmbeddings":
        embedding = CohereEmbeddings()

    # Retriever
    if args.retriever == "Chroma":
        vectorstore = Chroma.from_documents(qa_chunks, embedding)
        retriever = vectorstore.as_retriever()
    elif args.retriever == "TFIDFRetriever":
        retriever = TFIDFRetriever.from_documents(qa_chunks)
    elif args.retriever == "FAISS":
        vectorstore = FAISS.from_documents(qa_chunks, embedding)
        retriever = vectorstore.as_retriever()

    # LLM
    if "gpt" in args.llm:
        llm = ChatOpenAI(temperature=0, model_name=args.llm) # gpt-4, gpt-3.5-turbo, text-davinci-003
    elif "command" in args.llm:
        llm = Cohere(temperature=0, model=args.llm) # command, command-light
    elif "text" in args.llm:
        llm = OpenAI(temperature=0, model_name=args.llm)

    # Load up the prompt
    PROMPT = PromptTemplate.from_file(
        template_file=args.prompt_template_file,
        input_variables=["context", "question"]
    )

    # Setup QA Chain
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs
    )

    # Run through the questions
    predictions = []

    for idx, tmp_df in tqdm(eval_df.iterrows(), desc="Generating Answers"):
        question = tmp_df.question
        response = qa_chain.run(question)

        predictions.append({"response": response})

    # Set up LLM-based Evaluation
    qa_pairs = eval_df.to_dict("records")
    eval_chain = QAEvalChain.from_llm(llm = OpenAI(temperature=0))
    graded_outputs = eval_chain.evaluate(
        qa_pairs, predictions, question_key="question", prediction_key="response"
    )

    correct = 0
    for graded_output in graded_outputs:
        assert isinstance(graded_output, dict)
        if graded_output["text"].strip() == "CORRECT":
            correct+=1

    llm_accuracy = (correct/len(graded_outputs))*100
    wandb.log({"llm_based_eval_acc": llm_accuracy}, commit=False)

    # Setup `squad` metric evaluation
    squad_metric = load("squad")

    # Some data munging to get the examples in the right format
    for i, eg in enumerate(qa_pairs):
        eg["id"] = str(i)
        eg["answers"] = {"text": [eg["answer"]], "answer_start": [0]}
        predictions[i]["id"] = str(i)
        predictions[i]["prediction_text"] = predictions[i]["response"]

    for p in predictions:
        del p["response"]

    new_qa_pairs = qa_pairs.copy()
    for eg in new_qa_pairs:
        del eg["question"]
        del eg["answer"]
        del eg["model"]

    exact_matches = []
    f1s = []

    for qa_pair, prediction in zip(new_qa_pairs, predictions):
        result = squad_metric.compute(
            references=[qa_pair],
            predictions=[prediction],
        )
        exact_matches.append(result["exact_match"])
        f1s.append(result["f1"])

    mean_exact_match = np.mean(exact_matches)
    mean_f1 = np.mean(f1s)

    wandb.log({
        "exact_match": mean_exact_match,
        "f1": mean_f1
    }, commit=False)

    # Compile the whole evaluation in a single Table
    eval_df["Prediction"] = predictions
    eval_df["LLM Evaluating LLM"] = graded_outputs
    eval_df["Exact Match"] = exact_matches
    eval_df["F1"] = f1s

    wandb.log({"QA Eval Result": eval_df})


if __name__ == "__main__":
    args = get_args()
    print(args)

    main(args)