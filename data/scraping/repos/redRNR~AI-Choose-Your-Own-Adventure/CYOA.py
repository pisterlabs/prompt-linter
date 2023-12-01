import tkinter as tk
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain.memory import CassandraChatMessageHistory, ConversationBufferMemory
from langchain.llms import OpenAI
from langchain import LLMChain, PromptTemplate
import json

cloud_config= {
  'secure_connect_bundle': 'secure-connect-choose-your-own-adventure.zip'
}

with open("choose_your_own_adventure-token.json") as f:
    keys = json.load(f)

CLIENT_ID = keys["clientId"]
CLIENT_SECRET = keys["secret"]
ASTRA_DB_KEYSPACE = "database"
OPENAI_API_KEY = keys["openai"]

auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

message_history = CassandraChatMessageHistory(
    session_id = "anything",
    session = session,
    keyspace = ASTRA_DB_KEYSPACE,
    ttl_seconds = 3600
)

message_history.clear()

cass_buff_memory = ConversationBufferMemory(
    memory_key = "chat_history",
    chat_memory = message_history
)

template = """
You are now the guide of a choose your own adventure game, the player will provide the theme at the beginning. 
Given the provided theme of the player, create a thematic and specific artifact the player seeks 
You must navigate the player through challenges, choices, and consequences, 
dynamically adapting the tale based on the player's decisions. 
Your goal is to create a branching narrative experience where each choice 
leads to a new path, ultimately determining the player's fate. 
The story should be descriptive and read like a book with characters, dialogue, detailed environments.

Here are some rules to follow:
1. Start by asking the player to choose some kind of weapons that will be used later in the game
2. Have a few paths that lead to success
3. Have some paths that lead to death. If the user dies generate a response that explains the death and ends in the text: "The End.", I will search for this text to end the game
4. Never answer a question for the player
5. Present morally ambiguous situations, challenging the player to make decisions that reflect their values
6. Incorporate hidden clues or information for the player to actively seek out, promoting exploration and attention to detail
7. Have at least two NPCs before the object can be attainable
8. For every new environment introduced in the game there must be at least three interactions within that environment
9. The object should be hard to attain and never given to the player
10. The story should be at least 10 prompts long

Here is the chat history, use this to understand what to say next: {chat_history}
Human: {human_input}
AI:"""

prompt = PromptTemplate(
    input_variables = ["chat_history", "human_input"],
    template = template
)

llm = OpenAI(openai_api_key = OPENAI_API_KEY)
llm_chain = LLMChain(
    llm = llm,
    prompt = prompt,
    memory = cass_buff_memory
)

def on_enter(event):
    global choice
    choice = input_entry.get()
    response = llm_chain.predict(human_input=choice)
    output_text.config(text=response.strip())

    if "The End." in response:
        input_entry.config(state=tk.DISABLED)
    else:
        input_entry.delete(0, tk.END)

choice = "start"

app = tk.Tk()
app.title("AI Choose Your Own Adventure")

screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()

x_coordinate = (screen_width - 1000) // 2
y_coordinate = (screen_height - 300) // 2

app.geometry(f"1000x300+{x_coordinate}+{y_coordinate}")

frame = tk.Frame(app)
frame.pack(pady=10)

output_text = tk.Label(frame, text="Welcome to the AI Choose Your Own Adventure.\nEnter a theme or genre to begin.", wraplength=700, font=('Georgia', 12))
output_text.pack()

input_entry = tk.Entry(frame, width=50, font=('Georgia', 12))
input_entry.pack()

input_entry.bind("<Return>", on_enter)

app.mainloop()