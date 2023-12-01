import click
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline

from utilities.constants import CHROMA_CFG, CHROMA_PERSIST_DIR
from utilities.util import print_header

# @TODO Get from env file
model_id = "TheBloke/vicuna-7B-1.1-HF"
embedding_model = "hkunlp/instructor-xl"


def load_model():
    model, tokenizer = hw_model()

    p = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15
    )

    llm = HuggingFacePipeline(pipeline=p)

    return llm


def hw_model():
    """
    Load the correct setup for the given hardware
    :return: the model and tokenizer for the gpu or cpu
    """
    if torch.cuda.is_available():
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     )
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)
    return model, tokenizer


@click.command()
@click.option(
    "--device_type",
    default="cuda",
    type=click.Choice(
        [
            "cpu",
            "cuda",
        ]
    ),
    help="Hardware to use, cuda preferred",
)
def oasis_main(device_type):
    """
    Load the LLM and answer questions interactively
    :param device_type:
    :return:
    """
    qa = restore_llm(device_type)

    counter = 0
    # Interactive questions and answers
    while True:
        # Ask a question
        Q = input("Ask a question:")
        if Q == "exit":
            break

        # Get the answer
        A = qa(Q)
        answer, source = A["result"], A["source_documents"]
        counter += 1

        print_header(f"Question {counter}")
        print("Q: " + Q)
        print("A: " + answer)

        # Ask if user wants to see the source documents
        show_source = None
        while not (show_source == "y" or show_source == "n"):
            show_source = input("Show source documents? (y/n)")

            if show_source == "y":
                # # Print the relevant sources used for the answer
                # Testing print with tabbed lines
                for document in source:
                    print("\n> " + document.metadata["source"] + ":")
                    for line in document.page_content.split("\n"):
                        print("\t\t" + line)


def restore_llm(device_type):
    """
    Reload the LLM from the saved state and return the QA chain
    :param device_type:
    :return: A QA Chain pipeline that takes a question and returns an answer
    """

    # @TODO, add a progress bar that updates as each task is completed

    # Load the embedder
    print("Loading Embeddings...")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device_type}
    )
    # Load Chroma Vector Database
    print("Loading Chroma...")
    db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        client_settings=CHROMA_CFG,
    )
    golden = db.as_retriever()  # Who's a good retriever? You are!
    print("Loading LLM...")
    llm = load_model()
    # https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_qa.html
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=golden,
        return_source_documents=True
    )
    return qa


if __name__ == "__main__":
    oasis_main()
