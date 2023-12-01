import dotenv
import os
from datetime import date
from langchain.document_loaders import ObsidianLoader
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import weaviate


# Load and validate the config.
config = dotenv.dotenv_values()
required_keys = [
    "OPENAI_API_KEY",
    "OBSIDIAN_DATABASE_CLASS",
    "OBSIDIAN_VAULT_PATH",
    "WEAVIATE_URL",
    "WEAVIATE_API_KEY",
]
for key in required_keys:
    assert key in config, f"Key {key} not found in .env file"


def main():

    # Ask the user what to do.
    print("What do you want to do?")
    print("1. Create the database")
    print("2. Import from obsidian")
    print("3. Ask a question")
    choice = input("Enter your choice: ")

    if choice == "1":
        create_database()
    
    if choice == "2":
        vault_path = config["OBSIDIAN_VAULT_PATH"]
        import_vault(vault_path)

    if choice == "3":
        question = input("Enter your question: ")
        ask_question(question)


def create_database():
    client = get_client()

    class_object = {
        "class": config["OBSIDIAN_DATABASE_CLASS"],
        "description": "Notes from an Obsidian vault.",
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {
                "model": "ada",
                "modelVersion": "002",
                "type": "text"
            }
        },
    }

    try:
        client.schema.create_class(class_object)
        print(f"Created class {class_object['class']}")
    except Exception as e:
        print("Error creating class. Maybe it already exists?")
        print(e)


def import_vault(vault_path):

    assert os.path.exists(vault_path), f"PDF path does not exist: {vault_path}"

    # Load the obsidian documents.
    loader = ObsidianLoader(vault_path)
    documents = loader.load()

    # Create the client.
    client = get_client()

    # Get the database class.
    database_class = config["OBSIDIAN_DATABASE_CLASS"]

    # Import the data.  
    with client.batch(
            batch_size=100
        ) as batch:
            
            # Batch import the papers.
            for i, document in enumerate(documents):
                print(f"Importing document: {i+1}/{len(documents)}")

                properties = {
                    "source": document.metadata["source"],
                    "path": document.metadata["path"],
                    "created": str(date.fromtimestamp(document.metadata["created"])),
                    "last_modified": str(date.fromtimestamp(document.metadata["last_modified"])),
                    "last_accessed": str(date.fromtimestamp(document.metadata["last_accessed"])),
                    "page_content": document.page_content,
                }

                batch.add_data_object(
                    properties,
                    database_class,
                )
                

def ask_question(question, limit=5):

    # Get the client.
    client = get_client()

    nearText = {
        "concepts": [question]
    }

    database_class = config["OBSIDIAN_DATABASE_CLASS"]

    # Query the database.
    response = (
        client.query
        .get(database_class, ["source", "path", "created", "last_modified", "last_accessed", "page_content"])
        .with_near_text(nearText)
        .with_limit(limit)
        .do()
    )
    if "errors" in response:
        print(response["errors"])
        return

    # Get the results.
    results = response["data"]["Get"][database_class]
    print(f"Number of results: {len(results)}")


    # Create the prompt.
    prompt = ""
    for result in results:
        source = result["source"]
        path = result["path"]
        created = result["created"]
        last_modified = result["last_modified"]
        last_accessed = result["last_accessed"]
        page_content = result["page_content"]

        template_string = "This is note {source} at path {path} created at {created}, last modified at {last_modified}, last_accessed at {last_accessed}: {page_content}\n\n"
        page_prompt = PromptTemplate.from_template(template_string)
        page_prompt = page_prompt.format(source=source, path=path, created=created, last_modified=last_modified, last_accessed=last_accessed, page_content=page_content)
        prompt += page_prompt
    prompt += f"Please answer this question and tell me the path and the source: {question}"
    prompt = PromptTemplate.from_template(prompt)

    # Run the model.
    model = ChatOpenAI(
        openai_api_key=config["OPENAI_API_KEY"],
    )
    assert model is not None
    chain = LLMChain(llm=model, prompt=prompt, verbose=False)
    result = chain.run(text="")
    assert isinstance(result, str), f"result is not a string: {type(result)}"
    print(result)


def get_client():
    client = weaviate.Client(
        url = config["WEAVIATE_URL"],  
        auth_client_secret=weaviate.AuthApiKey(api_key=config["WEAVIATE_API_KEY"]), 
        additional_headers = {
            "X-Openai-Api-Key": config["OPENAI_API_KEY"],
        })
    return client


if __name__ == "__main__":
    main()