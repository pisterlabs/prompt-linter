# June 2023
# Nate Lalor

# imports
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain import OpenAI
import openai
import os


# this program takes in a txt or a pdf file and can acquire its content
# and return a txt summary of the data
# main function facilitates the function calls and prepares the data by splitting it up
def main():
    # load the dataset
    big_doc = load_data()

    # initializes API specifics
    llm, openai_ = llm_initialization()

    # TEMPORARY!!!!
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_)

    # setting up the function call
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

    # function call to split up the doc
    lg_docs = text_splitter.split_documents(big_doc)

    # get user input, to help focus prompts
    user_input = input(
        "What is the point of this deliverable? Please use specific language to specify what information should be included in the report. I want to focus on: "
    )
    print("Please wait while we produce the focused content...")

    # call the first execution
    summarized_log_doc = langchain_execution(llm, lg_docs, user_input)

    # and print the result: A summarized section of the original text
    print(summarized_log_doc)


# ----------------------------------------------------------- #


# load_data function prompts user input for filename
# then creates the sm_doc variable holding that loaded
# information ready to be manipulated
def load_data():
    filename = input("filename with extension: ")
    filename = filename.strip()
    print("Loading Data...")

    # this is so we can take txt or pdf file types
    sm_loader = UnstructuredFileLoader(filename)
    sm_doc = sm_loader.load()
    print("Data Loaded.")
    return sm_doc


# initializes the llm and OPENAI_API_KEY variables,
# basically preparing to use OpenAI's API
def llm_initialization():
    # LLM setup
    OPENAI_API_KEY = "sk-..."
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    return llm, OPENAI_API_KEY


# langchain_execution takes all the important information in: lg_docs,
# the set of split up text, llm, the language learning model (OpenAI),
# and the user's purpose/focus for this deliverable. It sets up prompts
# then makes a call to map_reduce chain through Langchain which produces
# our nice result
def langchain_execution(llm, lg_docs, user_input):
    # map prompt : given to produce each chunk
    map_prompt = (
        """
                 Write a concise summary focusing on %s:
                 "{text}"
                 CONCISE SUMMARY:
                 """
        % user_input
    )

    # make a PromptTemplate object using the s-string above
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    # combine prompt : the prompt it gives to "summarize", or how to sum up content into a final product.
    combine_prompt = (
        """Given the extracted content, create a detailed and thorough 3 paragraph report. 
                        The report should use the following extracted content and focus the content towards %s.
                        

                                EXTRACTED CONTENT:
                                {text}
                                YOUR REPORT:
                                """
        % user_input
    )

    # make a PromptTemplate object using the s-string above
    combine_prompt_template = PromptTemplate(
        template=combine_prompt, input_variables=["text"]
    )

    # line up all the data to our chain variable before the run execution below
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=False,
    )

    # execute the chain on the new split up doc
    summarized_log_doc = chain.run(lg_docs)

    # return the result
    return summarized_log_doc


if __name__ == "__main__":
    main()
