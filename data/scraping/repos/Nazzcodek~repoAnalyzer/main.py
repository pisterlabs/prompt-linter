import os
import pandas as pd
import nbformat
import chardet
import cohere
import streamlit as st
from nbconvert import PythonExporter
from github import Github
from langchain.prompts import PromptTemplate
from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere


GIT_TOKEN = os.environ.get("GIT_TOKEN")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
# GPT_TOKEN_LIMIT = 2048
MAX_LINE_LENGTH = 80
co = cohere.Client(COHERE_API_KEY)


def preprocess_file(file_path: str) -> list:
    """
    Preprocesses a file by splitting long lines of code into multiple lines.
    Args:
        file_path (str): The path of the file to preprocess.
    Returns:
        list: A list of code chunks where each chunk is a list of lines.
    """
    print(f"Processing file '{file_path}'")
    # Define file extensions to process
    code_extensions = [".py", ".java", ".cpp", ".js", ".c", ".html", ".css", ".rb"]  # Add more extensions as needed
    chunks = []
    
    if file_path.endswith(".ipynb"):
        # Process Jupyter Notebook file
        try:
            # Read notebook and create a PythonExporter
            notebook = nbformat.read(file_path, as_version=4)
            exporter = PythonExporter()
            
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    code = exporter.source_from_cell(cell)
                    # Split long lines of code into multiple lines
                    code_lines = []
                    for line in code.split("\n"):
                        if len(line) > MAX_LINE_LENGTH:
                            # Split the line into multiple lines
                            line_parts = [line[i:i+MAX_LINE_LENGTH] for i in range(0, len(line), MAX_LINE_LENGTH)]
                            code_lines.extend(line_parts)
                        else:
                            code_lines.append(line)
                    # Add the lines to the chunks list
                    chunks.extend(code_lines)
        except Exception as e:
            print(f"Error processing Jupyter Notebook file '{file_path}': {str(e)}")
    
    elif any(file_path.endswith(ext) for ext in code_extensions):
        # Process other code files
        try:
            # Detect encoding and read file
            with open(file_path, "rb") as f:
                result = chardet.detect(f.read())
                encoding = result["encoding"]

                with open(file_path, "r", encoding=encoding) as f:
                    for line in f:
                        if len(line) > MAX_LINE_LENGTH:
                            # Split the line into multiple lines
                            line_parts = [line[i:i+MAX_LINE_LENGTH] for i in range(0, len(line), MAX_LINE_LENGTH)]
                            chunks.extend(line_parts)
                        else:
                            chunks.append(line)
        except Exception as e:
            print(f"Error processing code file '{file_path}': {str(e)}")
    return chunks



def fetch_github_repos(username: str) -> list:
    """
    Fetches information about GitHub repositories for a given user.
    Args:
        username (str): The GitHub username to fetch repositories for.
    Returns:
        List[Dict[str, any]]: A list of dictionaries containing information about each repository.
    """
    # Initialize GitHub client
    client = Github(GIT_TOKEN)
    # Fetch user object
    user = client.get_user(username)
    # Fetch list of repositories
    repos = user.get_repos()
    # Initialize list to store repository information
    repo_info = []
    try:
        # Loop through each repository
        for repo in repos:
            try:
                # Fetch repository contents
                contents = repo.get_contents("")
                # Initialize list to store code contents for each file
                code_contents = []
                # Loop through each file in the repository
                for content in contents:
                    file_path = content.path
                    # Preprocess the file and extract code contents
                    chunks = preprocess_file(file_path)
                    code_contents.extend(chunks)
                # Add repository information to list
                repo_info.append({
                    "name": repo.name,
                    "description": repo.description,
                    "language": repo.language,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "labels": [label.name for label in repo.get_labels()],
                    "issues": repo.get_issues(state="all"),
                    "contents": code_contents,
                })
            except Exception as e:
                # Print error message if unable to fetch repository contents
                print(f"Error fetching contents of repository '{repo.name}': {str(e)}")
                continue
    except Exception as e:
        # Print error message if unable to fetch repositories
        print(f"Error fetching repositories: {str(e)}")
    # Return list of repository information
    return repo_info


# Define function with repo_info as input
def analyze_repos(repo_info):
    """
    Analyzes GitHub repositories to determine the most technically complex one.
    
    Args:
        repo_info (list of dict): A list of dictionaries containing information about GitHub repositories.
        
    Returns:
        dict: A dictionary containing the name and link of the most technically complex repository and an analysis of the repository.
    """
    # Create a DataFrame from the repo_info list and save it to a CSV file
    repo_info_df = pd.DataFrame(repo_info)
    repo_info_df.to_csv("repo_data.csv")
    # Load the CSV file and generate embeddings using Cohere API
    loader = CSVLoader(file_path="repo_data.csv", encoding="utf-8")
    csv_data = loader.load()
    csv_embeddings = CohereEmbeddings()
    vectors = Qdrant.from_documents(csv_data, csv_embeddings)
    # Define the context and prompt templates for the PromptSession
    context = """
    You are a Supersmart Github Repository AI system. You are a superintelligent AI that answers questions about Github Repositories and can understand the technical complexity of a repo.
    You have been asked to find the most technically complex and challenging repository from the given CSV file.
    To measure the technical complexity of a GitHub repository, you will analyze various factors such as the number of commits, branches, pull requests, issues, contents, number of forks, stars, and contributors. 
    Additionally, you will consider the programming languages used, the size of the codebase, and the frequency of updates.
    Calculate the complexity score for each project by assigning weights to each factor and summing up the weighted scores. The project with the highest complexity score will be considered the most technically complex.
    
    Analyze the following factors to determine the technical complexity of the codebase:
    1. Description
    2. Languages used in the repository
    3. Number of stars
    4. Number of forks
    5. Labels of the repository
    6. Issues associated with the repository
    7. Contents of the repository
    You can consider other factors as well if you think they are relevant for determining the technical complexity of a GitHub repository.
    Please provide a detailed analysis to justify your selection of the most technically complex repository.
    """
    prompt_template = """
    Understand the following to answer the question in an efficient way:
    {context}
    Question: {question}
    Now answer the question. Let's think step by step:
    """
    # Define the PromptSession and the RetrievalQA chain
    chain_type_kwargs = {"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])}
    chain = RetrievalQA.from_chain_type(llm=Cohere(cohere_api_key=COHERE_API_KEY, model="command-nightly", temperature=0),
                                        chain_type="stuff",
                                        retriever=vectors.as_retriever(),
                                        input_key="question",
                                        chain_type_kwargs=chain_type_kwargs)
    # Define the query and get the result from the RetrievalQA chain
    query = """
    Which is the most technically challenging repository from the given CSV file?
    Return
    the name of the repository, 
    the link to the repository, 
    and the analysis of the repository showing why it is the most technically challenging/complex repository. 
    Provide a detailed analysis to support your answer.
    The output should be in the following format:
    Repository Name: <name of the repository>
    Repository Link: <link to the repository>
    Analysis: <analysis of the repository>
    Provide a clickable link to the repository as well like this:
    [Repository Name](Repository Link)
    """

    result = chain({"question": query})
    return result


def main():
    """
    A function that analyzes GitHub repositories for technical difficulty.
    """
    # Set the Streamlit page configuration
    st.set_page_config(page_title="GitHub Repository Analyzer")
    
    # Set the page title
    st.title("Github Repository Analysis")
    # Get GitHub username from user input
    username = st.text_input("Enter a GitHub username to analyze:", value="")
    if username:
        # Fetch GitHub repositories and analyze them
        repo_info = fetch_github_repos(username)
        result = analyze_repos(repo_info)
        # Display the results
        st.header("Analysis Results")
        st.markdown(f"**GitHub Username:** {username}")
        st.markdown(f"**Most Technically Challenging Repository:** {result['name']}")
        st.markdown(f"**Repository Link:** [{result['name']}]({result['url']})")
        st.markdown(f"**Analysis:** {result['analysis']}")


if __name__ == '__main__':
    main()
