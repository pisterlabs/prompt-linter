import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI 
from langchain.chains import RetrievalQA
from langchain.llms import OpenAIChat
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

# Initialize Embeddings and Dataset
embeddings = OpenAIEmbeddings()
financial_report_dataset = 'hub://hollaugo/amazon_earnings_6'

#Initialize Retrieval Q&A
db = DeepLake(dataset_path=financial_report_dataset, embedding_function=embeddings, token=os.environ['ACTIVELOOP_TOKEN']) 
qa = RetrievalQA.from_chain_type(llm=OpenAIChat(model='gpt-4'), chain_type='stuff', retriever=db.as_retriever())


# Initializes your app with your bot token 
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

#handle message events with any text
@app.message("")
def message(message, say):
    response = qa.run(message["text"])
    say(response)




#Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()