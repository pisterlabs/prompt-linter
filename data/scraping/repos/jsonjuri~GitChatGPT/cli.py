# General modules
import argparse
import os
import sys
import time

# Modules
from dotenv import load_dotenv
from langchain.chains import RetrievalQA

from langchain.embeddings import (
    HuggingFaceEmbeddings,
    OpenAIEmbeddings
)

from langchain.prompts import PromptTemplate
from langchain.vectorstores import (
    Chroma,
    DeepLake
)

from common.databases.chroma import get as ChromaAdapter
from common.databases.deeplake import get as DeepLakeAdapter

# Utils
from common import config
from common.llm import (
    load_llm,
    set_llm
)

from common.git import (
    get_repository_url,
    get_repository,
    get_repository_name,
    remove_repository
)

# Rich
from rich.console import Console
console = Console()


if not load_dotenv():
    console.print(f"Could not load the .env file in the root of the project, or it is empty. Please check if it exists and is readable.", style="red")
    exit(1)


def get_user_choice(message: str, max_length: int):
    while True:
        model_choice = input(message)

        if model_choice.isdigit() and int(model_choice) <= max_length:
            return int(model_choice)
        else:
            console.print("Please enter a valid number.", style="yellow")


def main():
    config.display_intro()

    # If you want to add model llm to this list, you will need to follow these three steps:
    # 1) Create an env file in de /env directory
    # 2) Add the provider name to the list in .env file
    # 3) Add the necessary code to cli.py and common/embed.py
    # 4) Very important: Don't forget to do a pull request! (thank you!)
    llm_providers = config.get_list('LLM_PROVIDERS')
    console.print("LLM:", style="#a85ce6 bold underline")
    for i, option in enumerate(llm_providers, start=1):
        console.print(f"[{i}] - {option}", style="cyan")

    selected_model = get_user_choice("Choose your model LLM: ", len(llm_providers)) - 1

    # Switch the env file.
    config.switch_dotenv(selected_model)

    # If you want to add vector databases to this list, you will need to follow these three steps:
    # 1) Add the database name to the list in .env file
    # 2) Add the necessary code to cli.py and common/embed.py
    # 3) Very important: Don't forget to do a pull request! (thank you!)
    vector_databases = config.get_list('VECTOR_DB_PROVIDERS')
    if len(vector_databases) > 1:
        console.print("VECTOR DB:", style="#a85ce6 bold underline")
        console.print("[1] - Chroma", style="cyan")
        console.print("[2] - DeepLake", style="cyan")
        selected_database = get_user_choice("Choose your Vector Database: ", len(vector_databases)) - 1
    else:
        selected_database = 0

    # Override the vector db by the one from the user selection.
    os.environ['VECTOR_DB'] = config.get_list_selected('VECTOR_DB_PROVIDERS', selected_database)

    # Only needed when there are multiple database choices.
    if len(vector_databases) > 1:
        console.print(f"{config.get('VECTOR_DB')} has been set as your Vector Database.", style="green")
        print("")

    simple(
        selection=True
    )


def simple(selection):
    if selection is False:
        config.switch_dotenv(config.get('DEFAULT_LLM'))

        # Set the default Vector DB from config.
        os.environ['VECTOR_DB'] = config.get('DEFAULT_VECTOR_DB')
        console.print(f"{config.get('VECTOR_DB')} has been set as your Vector Database.", style="green")
        print("")

    console.print("EMBEDDINGS:", style="#a85ce6 bold underline")
    console.print("[1] - Github Repository", style="cyan")
    console.print("[2] - Documents", style="cyan")
    selected_embed_source = get_user_choice("Choose the type of data you would like to embed: ", 2)
    selected_embed_source_name = 'Github' if selected_embed_source == 1 else 'documents'
    console.print(f"{selected_embed_source_name} has been set as embeddings source.", style="green")
    print("")

    # Check and create directories if they do not exist.
    config.check_directories()

    # Skip the model directory check for OpenAI and Huggingface
    model = config.get('MODEL')

    # Set LLM variables.
    (llm_name, model_path, local_model) = set_llm(model, config.get('MODEL_TYPE'), config.get('MODEL_PATH'))

    # The base path of the Vector Database.
    db_base_path = os.path.join(config.get('DB_PATH'), str(config.get('VECTOR_DB')).lower())

    # At default in override mode, the selection means to override the current database.
    override_action = 2
    if selected_embed_source == 1:
        url = input("Enter the Github repository url: ")
        repository_url = get_repository_url(url)
        db_name = get_repository_name(repository_url)
        db_path = os.path.join(db_base_path, db_name)

        # If the vector database already exists we ask the user if he would like to continue or override the database.
        if os.path.exists(db_path):
            console.print("REPOSITORY ALREADY EXISTS:", style="#a85ce6 bold underline")
            console.print("[1] - Continue with the current database", style="cyan")
            console.print("[2] - Override the current database", style="cyan")
            override_action = get_user_choice("The repository database already exists what would you like to do: ", 2)
            print("")

            documents_path = get_repository(
                repository_url=repository_url,
                clone=False
            )

            # When the user choose for override then remove the repository and vector database.
            if override_action == 2:
                try:
                    remove_repository(db_path)
                except AssertionError as error:
                    console.print("An error occurred:", type(error).__name__, "–", error, style="red")
                    console.print("Make sure you have closed all terminals running GitChatGPT and have closed all connections to the database.", style="red")
                    exit(1)

                try:
                    remove_repository(documents_path)
                except AssertionError as error:
                    console.print("An error occurred:", type(error).__name__, "–", error, style="red")
                    console.print(
                        "Make sure you have closed all terminals running GitChatGPT and have closed all connections to the database.", style="red")
                    exit(1)

                print("")
                console.print("Cloning the repository 🚀💻", style="yellow")
                documents_path = get_repository(
                    repository_url=repository_url,
                    clone=True
                )
        else:
            console.print("Cloning the repository 🚀💻", style="yellow")
            documents_path = get_repository(
                repository_url=repository_url,
                clone=True
            )
    else:
        documents_path = 'documents'
        db_name = documents_path
        db_path = os.path.join(db_base_path, db_name)

    # Create embeddings.
    embeddings_kwargs = {'device': 'cuda'} if config.get('GPU_ENABLED') and llm_name != 'gpt4all' else {}

    if llm_name == 'openai' and config.get('OPENAI_EMBEDDINGS'):
        embeddings = OpenAIEmbeddings(
            api_key=config.get('OPENAI_API_KEY')
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.get('EMBEDDINGS_MODEL_NAME'),
            model_kwargs=embeddings_kwargs
        )

    match str(config.get('VECTOR_DB')).lower():
        case "chroma":
            db: Chroma = ChromaAdapter(
                override_action,
                documents_path,
                db_path,
                db_name,
                embeddings
            )

        case "deeplake":
            db: DeepLake = DeepLakeAdapter(
                override_action,
                documents_path,
                db_path,
                db_name,
                embeddings
            )

        case _default:
            # Raise an exception if the model_type is not supported
            console.print(f"Vector DB {config.get('VECTOR_STORE')} is not supported.", style="red")
            exit(1)

    # Delete the repository
    if selected_embed_source == 1:
        console.print("Deleting the repository 💣🚮 Adiós, old code! ", style="yellow")
        print("")
        #remove_repository(documents_path)

    # Parse the command line arguments
    args = parse_arguments()

    # Load Language Model (LLM)
    if local_model:
        console.print(f"Loading the model {model_path}", style="yellow")

    # GET LLM
    llm = load_llm(
        llm_name=llm_name,
        model=model,
        model_path=model_path,
        stream=args.mute_stream
    )

    # Set the data as retriever
    retriever = db.as_retriever(
        search_kwargs={"k": int(config.get('TARGET_SOURCE_CHUNKS'))}
    )

    # Retrieval QA
    prompt = PromptTemplate(
        template=config.prompt_template,
        input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": prompt} if config.get('PROMPT_ENABLED') else {}

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=not args.hide_source
    )

    print("")
    if selected_embed_source == 1:
        console.print("Ready to dive into the repository details. 🔍🚀 Fire away with your questions. 💬🔥", style="#eb6134")
    else:
        console.print("Ready to dive into the documents details. 🔍🚀 Fire away with your questions. 💬🔥", style="#eb6134")

    while True:
        # Ask for user input
        print("")
        question = input("What would you like to know about this repository?\n")
        print("")
        if question.lower().startswith("exit") or question.lower().startswith("quit"):
            print("Terminating program.")
            break
        if question.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        result = qa(question)
        answer, docs = result['result'], [] if args.hide_source else result['source_documents']
        end = time.time()

        # Print the result
        console.print("\n\n> Question:", style="magenta")
        console.print(question, style="green")
        console.print(f"\n> Answer (took {round(end - start, 2)} seconds):", style="#eb6134")
        console.print(answer.strip(), style="green")
        print("")

        # Print the relevant sources used for the answer
        # for document in docs:
        #    print(chalk.magenta("\n> Source: " + document.metadata["source"] + ":", style="#eb6134")
        #    print(chalk.magentaBright(document.page_content, style="#eb6134")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Engage with your github repository code through the power of LLM'
    )

    parser.add_argument(
        "--hide-source",
        "-S",
        action='store_true',
        help='Use this flag to disable the printing of source documents used for answers.'
    )

    parser.add_argument(
        "--mute-stream",
        "-M",
        action='store_true',
        help='Use this flag to disable the streaming StdOut callback for Language Models.'
    )

    return parser.parse_args()



# Main method
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("")
        console.print("\nI hope you found what you were looking for. 👀✨ See you next time! 👋🌟", style="#eb6134")
        print("")
        sys.exit(1)  # Use an exit code of your choice
    # except Exception as e:
    #    print("An error occurred:", e)
