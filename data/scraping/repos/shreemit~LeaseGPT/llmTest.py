from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
import pickle
import os
from dotenv import load_dotenv
from raw_strings import *
import openai


def get_listings_tool(retriever):
    tool_desc = '''Use this tool to inform user about listings from context. Give the user 2 options based on their criterion. If the user asks a question that is not in the listings, the tool will use OpenAI to generate a response.
    This tool can also be used for follow up quesitons from the user. 
    '''
    tool = Tool(
        func=retriever.run,
        description=tool_desc,
        name="Lease Listings Tool",   
    )
    return tool

def main():
    api_key = ""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    load_dotenv()

    if os.environ["OPENAI_API_KEY"] is not None:
        print("OPEN AI Key", os.environ["OPENAI_API_KEY"])
        # TODO: Scraping Craigslist

        # Combinig all the text into one string
        text = " ".join([doc1, doc2, doc3, doc4])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50, length_function=len
        )

        docs = [doc1, doc2, doc3, doc4]
        
        chunks = []

        # Splitting the text into chunks
        for doc in docs:
            if len(doc) > 1200:
                chunk_doc = text_splitter.split_text(doc)
                for chunk in chunk_doc:
                    chunks.append(chunk)
            else:
                chunks.append(doc)

        # st.write("Number of chunks", chunks)
        # chunks = text_splitter.split_text(text)

        store_name = "craigslist"
        template =  '''I want you to act to act like a leasing agent for me. Giving me the best options always based on what you read below. 
        You can give me something which matches my criteria or something which is close to it.
        '''
        # prompt = PromptTemplate(template=template, input_variables=["question"])


        # print("Prompt", prompt)

        embeddings = OpenAIEmbeddings()
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
                
        else:
            
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
                

        query = "Houses near UW"
        
        if query:
            llm=ChatOpenAI(
                openai_api_key=api_key,
                temperature=0,
                model_name='gpt-3.5-turbo'
            )

            retriever = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=VectorStore.as_retriever()
            )

            tools = [get_listings_tool(retriever=retriever)]
            memory = ConversationBufferWindowMemory(
                memory_key='chat_history',
                k=3,
                return_messages=True
            )

            conversational_agent = initialize_agent(
                agent='chat-conversational-react-description',
                tools=tools,
                llm=llm,
                verbose=True,
                max_iterations=2,
                early_stopping_method='generate',
                memory=memory
            ) 

            conversational_prompt = conversational_agent.agent.create_prompt(
                system_message = template,
                tools=tools,
            )

            conversational_agent.agent.llm_chain.prompt = conversational_prompt

            

    
            # print("agent", conversational_agent.agent.llm_chain.prompt)

            # print("messages", conversational_agent.agent.llm_chain.prompt.messages[2])
            # query= "Houses near UW"
            # prompt.format(question=query)
            # question = prompt.format(question=query)
            # print("Question", question)
            # st.write("Prompt", question)
            # st.write("Docs", docs)
            with get_openai_callback() as callback:
                # response = chain.run(input_documents=docs, question=question)
                # print("Response", response)
                # st.write(chain)
                op = conversational_agent("Give a few houses near UW")['output']
                print("Output", op)
                print("__________________________________")
                print("__________________________________")
                print("__________________________________")
                print("__________________________________")

                # print(conversational_agent("I have a budget of 2500"))
                print("Cost for query", callback.total_cost)
                
                #     st.write(response)
                # print("Response", response) 
        # except openai.error.AuthenticationError as e:
        #     # print("Error", e)
        #     print("Please enter a valid OpenAI API Key")
        # except:
        #     if os.environ["OPENAI_API_KEY"] is None:
        #         print("Please enter an OpenAI API Key")
        #     # if e == "No API key found":

if __name__ == "__main__":
    main()


