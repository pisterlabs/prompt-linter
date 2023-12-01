#https://platform.openai.com/docs/guides/embeddings/limitations-risks
#https://www.linkedin.com/pulse/build-qa-bot-over-private-data-openai-langchain-leo-wang/

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQAWithSourcesChain
#from main import display_menu
from dotenv import load_dotenv

from Modules.LLM import getKUKA_LLM
import logging
import ConsoleInterface

logger = logging.getLogger('ConsoleInterface')

load_dotenv()

#llm = ChatOpenAI(temperature=0)
llm=getKUKA_LLM()

embeddings = OpenAIEmbeddings()
embeddings.deployment = "kuka-text-embedding-ada-002"

persist_directory = "db"
vectorstore = None
collection_name = ""

def PDF2Chat_Run():

    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=input("Specify database name to load: "))
    
    logger.info("Vectorstore loaded successfully!\n")
    
    # Initialise Langchain - Conversation Retrieval Chain
    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True, verbose=True)
    
    
    # Front end web app
    import gradio as gr
    with gr.Blocks() as demo:
        Title = gr.Text("",label="Welcome to PDF2Chat!")
        chatbot = gr.Chatbot()
        msg = gr.Textbox("Enter your query here please!")
        
        #standardQuery = gr.Button("Ask me anything!")
        #openNewFile = gr.Button("Load PDF") 
        clear = gr.Button("Clear")
        chat_history = []

        def user(user_message, history):
            # Convert chat history to list of tuples
            
            history = [(msg[0], msg[1]) for msg in history]
        
            # Get response from QA chain
            response = qa({"question": user_message, "chat_history": history}, return_only_outputs=False)
            response['source_documents'][0]
            # Append user message and response to chat history as a tuple
            history.append((user_message, response["answer"]))


            return gr.update(value=""), history
        
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
        clear.click(lambda: None, None, chatbot, queue=False)
        #standardQuery.click(submitStandardQuery, [msg, chatbot], [msg, chatbot], queue=False)
        #standardQuery.click(submitStandardQuery, None, chatbot, queue=False)
        demo.launch(debug=True)
   
if __name__ == "__main__":
    PDF2Chat_Run()