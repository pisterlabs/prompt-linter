# Import necessary modules
from dotenv import load_dotenv
import os
import anthropic
import pprint
from halo import Halo
import tiktoken
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Load environment variables from .env file
load_dotenv()

# Initialize pretty printer for formatted printing
pp = pprint.PrettyPrinter(indent=4)

def generate_response(messages):
    """
    This function sends a completion request to the Claude model and returns the response.

    Parameters:
        messages (str): The conversation history.

    Returns:
        response: The response from the Claude model.
    """

    # Initialize spinner to show while waiting for response
    spinner = Halo(text='Loading...', spinner='dots')
    spinner.start()

    # Create an Anthropic client using the provided API key
    client = anthropic.Client(api_key=os.getenv("CLAUDE_KEY"))

    # Send a completion request to the Claude model
    response = client.completion(
        prompt=messages,
        stop_sequences = [anthropic.HUMAN_PROMPT],
        model=os.getenv("MODEL_NAME"),
        max_tokens_to_sample=1200,
    )

    # Stop the spinner after getting response
    spinner.stop()

    # Print the request and response
    print("Request:")
    pp.pprint(messages)
    print("Response:")
    pp.pprint(response)

    # Return the response
    return response

# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j

def main():
    """
    The main function to handle user input and interact with the Claude model.
    """

    patch_notes = ""
    # Open the file with 'read' permission
    with open('patch_notes.txt', 'r') as file:
        # Read the contents of the file into the patch_notes variable
        patch_notes = file.read()

    # Initialise tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    chunks = create_chunks(patch_notes,300,tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]

    chroma_client = chromadb.Client()
    embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_KEY"), model_name=os.getenv("EMBEDDING_MODEL"))
    collection = chroma_client.create_collection(name="conversations", embedding_function=embedding_function)

    index = 0
    for text_chunk in text_chunks:
        collection.add(
            documents=[text_chunk],
            ids=[f"text_{index}"]
        )
        index += 1
        

    # Initial conversation with Claude, giving it context about the Dota2 patch
    # messages = f"{anthropic.HUMAN_PROMPT} Please take this data on Dota2 latest patch, New Frontiers. You will refer to this data when I ask you about the latest patch of Dota2: {patch_notes}. {anthropic.AI_PROMPT} Understood"

    # Start a loop to continuously get user input and generate responses
    while True:

        # Get user input
        input_text = input("You: ")

        results = collection.query(
            query_texts=[input_text],
            n_results=20
        )

        knowledge_base = []
        for res in results['documents'][0]:
            knowledge_base.append(res)

        # print("Query results:")
        # pp.pprint(results)
        messages = f"{anthropic.HUMAN_PROMPT} Please take this data on Dota2 latest patch, New Frontiers. You will refer to this data when I ask you about the latest patch of Dota2: {knowledge_base}. {anthropic.AI_PROMPT} Understood"

        # Append the user's input to the conversation history
        messages = messages + f"{anthropic.HUMAN_PROMPT} {input_text} {anthropic.AI_PROMPT}"

        # Break the loop if user types "quit"
        if input_text.lower() == "quit":
            break

        # Generate a response from the Claude model
        response = generate_response(messages)
        # Append Claude's response to the conversation history
        messages = messages + response['completion']

        # Print the total number of tokens in the conversation history
        print(f"Token counts: {anthropic.count_tokens(messages)}")

        # Print Claude's response
        print(f"Claude: {response['completion']}")

# Run the main function if the script is run directly
if __name__ == "__main__":
    main()