import os
from langchain.document_loaders.base import BaseLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
import src.ai_outputparsers as parsers
import logging

# set logger
logger = logging.getLogger(__name__)

class CustomCodeLoader(BaseLoader):
    """
    Custom loader for loading code files into documents, handling various programming languages.

    Attributes:
    - files (dict): A dictionary where keys are file paths and values are file contents.
    """

    # Mapping of file extensions to their corresponding languages
    extension_to_language = {
        '.py': 'python',
        '.ipynb': 'jupyter notebook',
        '.r': 'R',
        '.js': 'javascript',
        # Add more mappings as needed
    }

    def __init__(self, files):
        """
        Initializes the CustomCodeLoader with a set of files.

        Parameters:
        - files (dict): A dictionary of files with paths as keys and contents as values.
        """
        self.files = files

    def get_language_from_extension(self, file_path):
        """
        Determines the programming language of a file based on its extension.

        Parameters:
        - file_path (str): The path of the file.

        Returns:
        str: The detected programming language, or 'unknown' if not identifiable.
        """
        _, extension = os.path.splitext(file_path)
        return self.extension_to_language.get(extension.lower(), 'unknown')

    def load(self):
        """
        Iterates over the provided files, creating a document structure for each.

        Yields:
        dict: A document containing the file content and metadata, including the file path and detected language.
        """
        for path, content in self.files.items():
            if isinstance(content, str):
                language = self.get_language_from_extension(path)
                document = {
                    'page_content': content,
                    'metadata': {
                        'path': path,
                        'language': language
                    }
                }
                yield document

    
def get_repo_overview(documents, openai_api_key):
    """
    Generates a brief overview of a GitHub repository suitable for a README file.

    Parameters:
    - documents (list): A list of documents representing the contents of the repository.
    - openai_api_key (str): The API key for OpenAI's GPT model.

    Returns:
    str: A paragraph summarizing the purpose and contents of the repository.
    """
   
    #template = """
    #You are a helpful assistant who generates summarizations of code to build a README file. 
    #Give me a one paragraph with a brief overview of what is the repo for. 
    #"""
    #human_template = "Repo content: {documents}"
    template = """
    You are a helpful assistant who helps to build a README file. You will receive a list containing
    the summaries of each file on the repository. Using this information, summarize the general purpose
    of the project and the tools used in one paragraph.  
    """
    human_template = "File summaries: {documents}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    
    chain = chat_prompt | ChatOpenAI(openai_api_key = openai_api_key) | parsers.FormattedOutputConvertToText()
    answer = chain.invoke({"documents": documents})

    return(answer)


def get_repo_structure(documents, openai_api_key):
    """
    Generates a tree structure representation of the repository's file structure.

    Parameters:
    - documents (list): A list of documents representing the contents of the repository.
    - openai_api_key (str): The API key for OpenAI's GPT model.

    Returns:
    str: A markdown-formatted tree structure of the repository's file system.
    """    

    template = """
    You are a helpful assistant who helps to build a README file. 
    Return your answer as a tree generator with no further explanations. 
    """

    human_template = """
    This are the list of paths of the reposoitory files: {list_of_paths}.
    Give me the repo structure.
    """

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    chain = chat_prompt | ChatOpenAI(openai_api_key=openai_api_key) | parsers.MarkdownTreeStructureOutputParser()
    #answer = chain.invoke({"list_of_paths": [document["metadata"]["path"] for document in documents]})
    answer = chain.invoke({"list_of_paths": documents})

    return(answer)


def getting_started(repo_name, documents, openai_api_key):
    """
    Provides instructions for getting started with a GitHub repository, including 
    cloning and installing dependencies.

    Parameters:
    - repo_name (str): The name of the repository.
    - documents (list): A list of documents representing the contents of the repository.
    - openai_api_key (str): The API key for OpenAI's GPT model.

    Returns:
    str: Step-by-step instructions for getting started with the repository.
    """

    #template = """
    #You are a helpful assistant who helps built a README file for a Github repository. 
    #Give me instructions on how to to get started. detailing the steps for cloning the repository 
    #and the steps for installing dependencies. 
    #"""
    template = """
    You are a helpful assistant who helps built a README file for a Github repository. 
    Give me instructions on how to to get started. detailing the steps for cloning the repository, 
    the steps for installing dependencies, and any special credentials needed. 
    """

    human_template = """
    Repo Name: {repo}. 
    List of files: {file_paths}. 
    Content of files: {contents}
    """

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    chain = chat_prompt | ChatOpenAI(openai_api_key = openai_api_key) | parsers.FormattedOutputConvertToText()
    answer = chain.invoke({"repo": repo_name, "file_paths": [doc["metadata"]["path"] for doc in documents],"contents": [doc["page_content"] for doc in documents]})

    return answer

def cloning_instructions(repo_name, paths, openai_api_key):
    """
    Provides instructions for cloning a GitHub repository to get started.

    Parameters:
    - repo_name (str): The name of the repository.
    - paths (list): A list of documents representing the contents of the repository.
    - openai_api_key (str): The API key for OpenAI's GPT model.

    Returns:
    str: Instructions for cloning the repository.
    """

    template = """
    You are a helpful assistant who helps built a README file for a Github repository. 
    Give me instructions on how to clone the repository and install dependencies. After your
    reponse append this text 'NOTE: The Github repo is too long to give more detailed instructions. 
    Parsing of code is not implemented yet.' 
    """

    human_template = """
    Repo Name: {repo}. 
    List of files: {file_paths}. 
    """

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    chain = chat_prompt | ChatOpenAI(openai_api_key = openai_api_key) | parsers.FormattedOutputConvertToText()
    answer = chain.invoke({"repo": repo_name, "file_paths": paths})

    return answer


def get_file_summaries(documents, openai_api_key):
    """
    Generates summaries for each file in the repository.

    Parameters:
    - documents (list): A list of documents, each containing file path and content.
    - openai_api_key (str): The API key for OpenAI's GPT model.

    Returns:
    list: A list of summaries for each file in the repository.
    """
    
    template = """
    You are a helpful assistant who generates summarizations of files to build a README file. 
    Return your response in the format File Path: the_file_path Summary: the_summary
    """
    human_template = "The file {path} has this content: {content}"
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    chain = chat_prompt | ChatOpenAI(openai_api_key = openai_api_key) | parsers.FormattedOutputParserSummary()

    # List of summaries
    summaries = []

    for document in documents:
        # Extract the path and code content from each document
        path = document["metadata"]["path"]
        content = document["page_content"]

        if path in [".DS_Store", ".gitignore", ".github/workflows/main.yaml"]:
            logger.info("Skipping summarization of file %s", path)
            continue
        else:
            logger.info("Summarizing file %s", path)
            # Invoke the chain for each document
            response = chain.invoke({"path": path, "content": content})

        # Append the generated summary to the summaries list
        summaries.append(response)

    return summaries
        