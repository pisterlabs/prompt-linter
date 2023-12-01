import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import JiraToolkit
from langchain.llms import OpenAI
from langchain.utilities.jira import JiraAPIWrapper
import curses
import readline

load_dotenv(override=True)
OPEN_AI_TOKEN=os.getenv("OPENAI_API_KEY")
ACTIVELOOP_TOKEN=os.getenv("ACTIVELOOP_TOKEN")
JIRA_API_TOKEN=os.getenv("JIRA_API_TOKEN")
JIRA_INSTANCE_URL=os.getenv("JIRA_INSTANCE_URL")
JIRA_USERNAME=os.getenv("JIRA_USERNAME")

print("ENV VARIABE", OPEN_AI_TOKEN)
print("ENV VARIABE", ACTIVELOOP_TOKEN)
print("ENV VARIABE", JIRA_API_TOKEN)
print("ENV VARIABE", JIRA_INSTANCE_URL)
print("ENV VARIABE", JIRA_USERNAME)
embeddings = OpenAIEmbeddings()

db = DeepLake(
    dataset_path="hub://variablevasas/syndicate-content-sync-v2",
    read_only=True,
    embedding_function=embeddings,
)
retriever = db.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["fetch_k"] = 100
retriever.search_kwargs["maximal_marginal_relevance"] = True
retriever.search_kwargs["score_threshold"] = 0.9
retriever.search_kwargs["k"] = 15

def filter(x):
    # filter based on source code
    if "com.google" in x["text"].data()["value"]:
        return False

    # filter based on path e.g. extension
    metadata = x["metadata"].data()["value"]
    print(metadata["source"])
    return "go" in metadata["source"] or "mod" in metadata["source"]


model = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613", max_tokens=6000)  # switch to 'gpt-4'
qa = ConversationalRetrievalChain.from_llm(
    model, 
    chain_type='refine',
    retriever=retriever, 
    verbose=True,
    condense_question_llm = ChatOpenAI(temperature=0.3, model='gpt-3.5-turbo-16k-0613'),
)

chat_history = []
queries = []

def main():
    while True:
        i = len(queries)-1
        try:
            query = input("Query: ")
        except EOFError:
            # Handle Ctrl+D
            print("\nDo you want to exit? (yes/no)")
            choice = input().lower()
            if choice == 'yes':
                print("Bye!")
                break
        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\nDo you want to exit? (yes/no)")
            choice = input().lower()
            if choice == 'yes':
                print("Bye!")
                break

        try:
            result = qa({"question": query, "chat_history": chat_history})
        except Exception as e:
            print(e)
            continue
        print(result["answer"])

    # Add the input to the history
    chat_history.append((query, result["answer"]))
    readline.add_history(query)

if __name__ == "__main__":
    main()