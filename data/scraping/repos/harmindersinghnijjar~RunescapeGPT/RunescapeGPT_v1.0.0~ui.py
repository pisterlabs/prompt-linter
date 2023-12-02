import openai
import os
import flet as ft
from flet import Page, TextField, ElevatedButton, ListView, Text
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the custom dataset
loader = DirectoryLoader("mydata/")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Embed the documents and store them in a vector database
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
docsearch = Chroma.from_documents(texts, embeddings)

# Create the ConversationalRetrievalChain
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
)


# Define the main Flet app UI and interaction
def main(page: Page):
    page.title = "Realtime Chat with OpenAI"
    page.update()

    # UI elements
    chat_history_list = ListView(auto_scroll=True)
    message_input = TextField(
        hint_text="Type your message here", autofocus=True, on_submit=send_message
    )
    send_button = ElevatedButton(text="Send", on_click=send_message)
    message_input.focus()

    # Layout
    page.add(chat_history_list, message_input, send_button)

    # Function to handle sending messages
    def send_message(e):
        query = message_input.value.strip()
        if not query:
            # No input provided
            return

        # Simulate user message in chat history
        chat_history_list.items.append(Text(value=f"You: {query}"))
        chat_history_list.update()

        # Call the chain with the user's query and chat history
        result = chain(
            {"question": query, "chat_history": page.data.get("chat_history", [])}
        )
        answer = result["answer"]

        # Simulate bot response in chat history
        chat_history_list.items.append(Text(value=f"Bot: {answer}"))
        chat_history_list.update()

        # Append to chat history and maintain its size
        chat_history = page.data.get("chat_history", [])
        chat_history.append((query, answer))
        if len(chat_history) > max_history:
            chat_history.pop(0)
        page.data["chat_history"] = chat_history

        # Clear input field
        message_input.value = ""
        message_input.update()

    # Set the max history for chat to keep
    max_history = 5
    page.data["chat_history"] = []


if __name__ == "__main__":
    ft.app(target=main, host="127.0.0.1", port=5000)
