# Import necessary libraries
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
from langchain.schema.output_parser import StrOutputParser
from utils import build_chat_chain, run_hf_chain
import time
import datetime as dt
import json
import numpy as np
from pymongo import MongoClient


# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    
# Store cumulative cost of interaction
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0

# Store current query (i.e., last user input)
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
    
# Store the time of first interaction (when they opened the app) 
if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = time.time()
    
# Store cumulative time spent over the session.
if "time_spent" not in st.session_state:
    st.session_state.time_spent = 0
    
# callback to record time until feedback
def record_time_until_feedback():
    st.session_state.time_until_feedback = time.time() - st.session_state.session_start_time
    
# This function will run after each interaction (i.e., clicks any button) to keep track of total seconds spent in the session.
# I don't think is the best way to do it, but it works as a proxy for now. 
def acumulate_time():
    st.session_state.time_spent += time.time() - st.session_state.session_start_time
    st.session_state.session_start_time = time.time()
    
    
# Store feedback
if "feedback" not in st.session_state:
    st.session_state.feedback = ""        
    
# callback to store feedback in text file
def record_feedback():
    with open("feedback.txt", "a") as f:
        f.write(f"FEEDBACK: \n\n {st.session_state.feedback} \n\n ------------------ \n\n SUBMISSION DATE: {dt.datetime.today()} \n")
        f.write("\n")

# callback to store feedback, submission date, chat history, and total cost in a json file
def record_feedback_json():
    dict_feedback = {"Random_id": np.random.randint(10000),
                     "Content": {
                                "feedback": st.session_state.feedback,
                                "chat_history": st.session_state.messages,
                                "total_cost": st.session_state.total_cost,
                                "time_until_feedback": st.session_state.time_spent,
                                "submission_date": dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                                }
                     }
    
    json_feedback = json.dumps(dict_feedback, indent=2)
    
    with open("feedback.json", "a") as f:
        f.write(json_feedback)
        f.write("\n")
        
    st.sidebar.write("Feedback stored in feedback.json")
    
def record_feedback_mongo():
    uri = f"mongodb+srv://{st.secrets['mongo']['username']}:{st.secrets['mongo']['password']}@silicusta.jfzl5zt.mongodb.net/?retryWrites=true&w=majority&appName=AtlasApp"

    # Create a new client and connect to the server
    client = MongoClient(uri)

    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!") 
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        # database and collection code goes here
        db = client.Feedback_DB
        #st.sidebar.write(f"Collection names: {db.list_collection_names()}")
        coll = db.feedback
        
        
        # insert code goes here
        feedback_doc = {"Random_id": np.random.randint(10000),
                        "Content": {
                                    "feedback": st.session_state.feedback,
                                    "chat_history": st.session_state.messages,
                                    "total_cost": st.session_state.total_cost,
                                    "time_until_feedback": st.session_state.time_spent,
                                    "submission_date": dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                        }

        coll.insert_one(feedback_doc)
        print(f"Inserted feedback in MongoDB! \n {feedback_doc}")
            
        # Close the connection to MongoDB when you're done.
        client.close()
        print("Successfully stored feedback in MongoDB!")
        st.sidebar.write("Feedback stored in MongoDB!")
    except Exception as e:
        print(e)
        # Close the connection to MongoDB when you're done.
        client.close()
        st.sidebar.write(f"Failed to store feedback in MongoDB with error: {e}")

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# -------------------------------- #

def run_silicus_ta():
    st.set_page_config(page_title="🤗💬 Silicus TA")
    
    chain, vectorstore = build_chat_chain()
    
    with st.sidebar:
        st.title('🤗💬 Silicus TA')
        st.subheader('Powered by 🤗 Language Models')
        #system_prompt = st.text_area("Enter your system prompt here. This will tune the style of output (kind of like the persona of the model).", height=150)
        
        st.sidebar.subheader("Feedback")
        st.sidebar.write("Was the conversation helpful? Your honest feedback will help me improve the system.")
        feedback = st.sidebar.text_area("Feedback", height=150)
        if st.sidebar.button("Submit Feedback", on_click=record_time_until_feedback):
            st.session_state.feedback = feedback
            st.sidebar.success("Thank you for your feedback!")
            #record_feedback()
            #record_feedback_json()
            record_feedback_mongo()
        st.sidebar.write(f"Cumulative time spent: {round(st.session_state.time_spent, 2)} seconds")
            
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
                

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    # User-provided prompt
    # We instantiate a new prompt with each chat input
    if prompt := st.chat_input(on_submit=acumulate_time):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            st.session_state.current_query = prompt

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                answer, cost = run_hf_chain(st.session_state.messages,
                                            st.session_state.current_query, 
                                            chain, vectorstore)
            
                
                st.write(answer["text"])
                st.session_state.total_cost += cost
                st.sidebar.write(f"Cost of interaction: {cost}")
                st.sidebar.write(f"Total cost: {st.session_state.total_cost}")
        message = {"role": "assistant", "content": answer["text"]}
        st.session_state.messages.append(message)
    
if __name__ == "__main__":
    response_start_time = time.time()
    run_silicus_ta()
    response_end_time = time.time()

    print(f"Response time: {response_end_time - response_start_time}")
    
    