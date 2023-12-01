from datetime import datetime
import os
import singlestoredb as s2
import pyaudio
import wave
import json
import pandas as pd
import numpy as np
import openai
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import Canvas
from tkinter import Label
from sqlalchemy import create_engine
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]

#  SS API Key : 6b3c9c2fa918cff89da47cdb95849ffecb909104aee2d464da03f7c06d7ccf40
user = 'admin'
password = 'Password123!'
host = 'svc-39644b79-076a-44e6-8b7f-d6415f18d4c8-dml.aws—virginia-6.svc.singlestore.com'
port = 3306
database = 'ai_demo'
table_name = 'embeddings'
model = 'text-embedding-ada-002'


# Create the agent executor
db = SQLDatabase.from_uri(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}", include_tables=['embeddings', 'stock_table'], sample_rows_in_table_info=1)
print(db)
llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0, verbose=True)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True,
    prefix= '''
    You are an agent designed to interact with a SQL database called SingleStore. This sometimes has Shard and Sort keys in the table schemas, which you can ignore. 
    \nGiven an input question, create a syntactically correct MySQL query to run, then look at the results of the query and return the answer. 
    \n If you are asked about similarity questions, you should use the DOT_PRODUCT function.
    
    \nHere are a few examples of how to use the DOT_PRODUCT function:
    \nExample 1:
    Q: how similar are the questions and answers?
    A: The query used to find this is:
    
        select question, answer, dot_product(question_embedding, answer_embedding) as similarity from embeddings;
        
    \nExample 2:
    Q: What are the most similar questions in the embeddings table, not including itself?
    A: The query used to find this answer is:
    
        SELECT q1.question as question1, q2.question as question2, DOT_PRODUCT(q1.question_embedding, q2.question_embedding) :> float as score
        FROM embeddings q1, embeddings q2 
        WHERE question1 != question2 
        ORDER BY score DESC LIMIT 5;
    
    \nExample 3:
    Q: In the embeddings table, which rows are from the chatbot?
    A: The query used to find this answer is:
    
        SELECT category, question, answer FROM embeddings
        WHERE category = 'chatbot';
        
    \nUnless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    \n The question embeddings and answer embeddings are very long, so do not show them unless specifically asked to.
    \nYou can order the results by a relevant column to return the most interesting examples in the database.
    \nNever query for all the columns from a specific table, only ask for the relevant columns given the question.
    \nYou have access to tools for interacting with the database.\nOnly use the below tools. 
    Only use the information returned by the below tools to construct your final answer.
    \nYou MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again up to 3 times.
    \n\nDO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    \n\nIf the question does not seem related to the database, just return "I don\'t know" as the answer.\n,
    
    ''',
    format_instructions='''Use the following format:\n
    \nQuestion: the input question you must answer
    \nThought: you should always think about what to do
    \nAction: the action to take, should be one of [{tool_names}]
    \nAction Input: the input to the action
    \nObservation: the result of the action
    \n... (this Thought/Action/Action Input/Observation can repeat 10 times)
    \nThought: I now know the final answer
    \nFinal Answer: the final answer to the original input question\n
    \n\nSQL Query used to get the Answer: the final sql query used for the final answer',
    ''',
    top_k=5,
    max_iterations=10
)

# User Interface Creation 

root = tk.Tk()
root.geometry("750x570")
# root.geometry("500x380")
root.title("Denjiro（⚔️—⚔️）")

# Labels
name = Label(root, text="Query", font=("Arial", 22)).place(x=30,y=30)
response = Label(root, text="Chatbot response").place(x=30,y=290)
# name = Label(root, text="Question").place(x=20,y=20)
# response = Label(root, text="Chatbot response").place(x=20,y=160)

# Create the text entry widget
entry = ttk.Entry(root, font=("Arial", 22))
entry.pack(padx=30, pady=75, fill=tk.X)
# entry = ttk.Entry(root, font=("Arial", 14))
# entry.pack(padx=20, pady=50, fill=tk.X)
entry.insert(0, f"Enter your database {database} query here")

# get embedding functions
def get_embedding(text, model=model):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def insert_embedding(question):
    category = 'chatbot'
    question_embedding = get_embedding(question, model=model)
    answer = agent_executor.run(question)
    answer_embedding = get_embedding(answer, model=model)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # add questions and answer embeddings to a dataframe
    df = pd.DataFrame(columns=['category','question','question_embedding','answer', 'answer_embedding', 'created_at'])
    new_row = {'category':category, 'question':question, 'question_embedding':question_embedding,'answer':answer, 'answer_embedding':answer_embedding, 'created_at':created_at }
    df = df.append(new_row, ignore_index=True)
    # print(df['answer'])
    
    # send to SingleStore
    mystmt = "INSERT INTO {} (category, question, question_embedding, answer, answer_embedding, created_at) VALUES ('{}',\n'{}', \njson_array_pack('{}'), \n'{}', \njson_array_pack('{}'), \n'{}')"


    for i in range(len(df)):
        stmt = mystmt.format(table_name, df['category'][i],df['question'][i].replace("'",""), df['question_embedding'][i], df['answer'][i].replace("'",""), df['answer_embedding'][i], df['created_at'][i])
        
        
        # executable_stmt = text(stmt)
        engine = s2.connect(host=host, port=port, user=user, password=password, database=database)

        with engine:
            with engine.cursor() as cur:

                cur.execute(stmt)
                for row in cur.fetchall():
                    print(row)
                    cur.close()    

# Create the button callback
def on_click():
    # Get the query text from the entry widget
    query = entry.get()

    # Run the query using the agent executor
    result = agent_executor.run(query)

    # Display the result in the text widget
    text.delete("1.0", tk.END)
    text.insert(tk.END, result)

    # get result embedding
    result_embedding = get_embedding(result)
    insert_embedding(query)

# Create the clear button callback
def clear_text():
    text.delete("1.0", tk.END)

    # Clear the entry widget
    entry.delete(0, tk.END)
    entry.insert(0, f"Enter your quesion on database: {database}")

# Create noise gate
def apply_noise_gate(audio_data, threshold):
    # Calculate the root mean square (RMS) of the audio data
    valid_data = np.nan_to_num(audio_data, nan=0.0)

    # valid_data = ...  # Your valid data here

    # Compute the square of valid_data
    squared_data = np.square(valid_data)

    # Check for negative or invalid values
    invalid_indices = np.isnan(squared_data) | (squared_data < 0)

    # Set negative or invalid values to 0
    squared_data[invalid_indices] = 0

    # Compute the mean of squared_data
    mean_squared = np.mean(squared_data)

    # Compute the root mean square (RMS)
    rms = np.sqrt(mean_squared)

    # Check if the RMS value is a valid number
    if np.isnan(rms):
        return audio_data
    
    # If RMS is below the threshold, set all samples to zero
    if rms < threshold:
        audio_data = np.zeros_like(audio_data)

    return audio_data

# Create the mic button callback
def record_audio(output_file, sample_rate=44100, chunk_size=1024, audio_format=pyaudio.paInt16, channels=1, threshold=0.01):
    audio = pyaudio.PyAudio()

    print('say something') 
    # replace with beep?
    
    # Open the microphone stream
    stream = audio.open(format=audio_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    frames = []
    silence_frames = 0
    silence_threshold = 80.01  # Adjust this value according to your environment

    # Record audio until there is 2 seconds of silence
    while True:
        data = stream.read(chunk_size)
        frames.append(data)

        # Convert data to numpy array for analysis
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Apply noise gate to reduce background noise
        audio_data = apply_noise_gate(audio_data, threshold)

        # Check if the audio is silent (below the threshold)
        if np.max(np.abs(audio_data)) < silence_threshold:
            silence_frames += 1
        else:
            silence_frames = 0

        # Break the loop if there is 2 seconds of silence
        if silence_frames / (sample_rate / chunk_size) >= 2:
            break

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio as a WAV file
    wave_file = wave.open(output_file, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(audio.get_sample_size(audio_format))
    wave_file.setframerate(sample_rate)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

def transcribe_mic():
    # Usage
    output_file = 'recording.wav'
    record_audio(output_file)
    print(f"Recording saved as {output_file}")

    audio_file= open("recording.wav", "rb")
    
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    entry.delete(0,tk.END)
    entry.insert(0, transcript["text"])

    on_click()

def mic_button_actions():
    ttk.Button(root, text="Mic", command=transcribe_mic).place(x=45, y=150)
    # ttk.Button(root, text="Mic", command=transcribe_mic).place(x=30, y=100)
    new_embedding = get_embedding(transcript)
    print(new_embedding)

# Create the mic button widget
# Create a style with the desired font settings
style = ttk.Style()
style.configure("TButton", font=("Arial", 22))

mic_button = ttk.Button(root, text="Mic", command=transcribe_mic).place(x=45, y=150)
# mic_button = ttk.Button(root, text="Mic", command=transcribe_mic).place(x=30, y=100)

# Create the button widget
button = ttk.Button(root, text="Chat", command=on_click).place(x=225, y=150)
# button = ttk.Button(root, text="Chat", command=on_click).place(x=150, y=100)

# Create the clear button widget 
clear_button = ttk.Button(root, text="Reset", command=clear_text).place(x=405, y=150)
# clear_button = ttk.Button(root, text="Reset", command=clear_text).place(x=270, y=100)

# Create the text widget to display the result
text = tk.Text(root, height=15, width=90, font=("Arial", 22))
# text = tk.Text(root, height=10, width=60, font=("Arial", 14))
text.pack(side=tk.BOTTOM, padx=20, pady=20)

# Start the UI event loop
root.mainloop() 