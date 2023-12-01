import sys
import time
from dotenv import load_dotenv
import pickle
import os

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')


from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class Tee(object):
    def __init__(self, terminal, file, start_word_count=0):
        self.terminal = terminal
        self.file = file
        self.word_count = start_word_count

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.word_count += len(message.split())
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()



# Check existing word count in the file
def get_existing_word_count(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r') as f:
        text = f.read()
        return len(text.split())


file_path = 'output.txt'
existing_word_count = get_existing_word_count(file_path)
# Start capturing terminal output to both terminal and 'output.txt' file.
file = open(file_path, 'a')
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, file, start_word_count=existing_word_count)

# Define shared context here:
all_header = """
Roll for Shoes: Cyberpunk Edition

In the neon-soaked streets of 2020, amidst the towering megacorps and digital shadows, players navigate the matrix of intrigue and tech.

1. State your action and roll a number of D6s, based on the relevant skill's level.
2. If your roll's sum is higher than the challenge or opponent's roll, your action succeeds.
5. Gain 1 XP for failed attempts. The neon world is harsh, but every failure is a lesson.
6. Use XP to make one die a 6, but only at the GMs allowance to further the story. 
Characters evolve based on choices, leading to unexpected talents in this cybernetic sprawl.
"""
#class SummarizationAgent:
#    def __init__(self):
#        self.chat = ChatOpenAI(streaming=True,
#                               callbacks=[StreamingStdOutCallbackHandler()],
#                               temperature=0.7,
#                               model="gpt-4",)

#    def summarize(self, conversation):
#        summary_prompt = f"Please provide a truncation of the following conversation: {conversation}. Ensure that the truncation has the same style and feel as the content, so as not to feel like a summary."
#        resp = self.chat([HumanMessage(content=summary_prompt)])
#        return resp.content



class Agent:
    def __init__(self, name, prompt, history_file=None):
        self.name = name
        self.history_file = history_file
        self.chat = ChatOpenAI(streaming=True,
                               callbacks=[StreamingStdOutCallbackHandler()],
                               temperature=1.0,
                               model="gpt-4",)
        self.message_history = self.load_conversation() if history_file else []
        self.message_history.insert(0, SystemMessage(content=prompt))

    def message(self, message):
        self.message_history.append(HumanMessage(content=message))
        resp = self.chat(self.message_history)
        print("\n")
        self.message_history.append(resp)
        self.save_conversation()
        return resp

    def save_conversation(self):
        if not self.history_file:
            return
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(self.message_history, f)
            #print(f"Conversation saved to {self.history_file}")
        except Exception as e:
            print(f"Error saving conversation: {e}")

    def load_conversation(self):
        if not self.history_file or not os.path.exists(self.history_file):
            return []
        try:
            with open(self.history_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return []
    def word_count(self):
        return sum(len(message.content.split()) for message in self.message_history)
        
def save_to_text_file(filename):
    with open(filename, 'w') as f:
        for agent in [gm, player]:
            for message in agent.message_history:
                f.write(f"{agent.name}: {message.content}\n")


TARGET_WORD_COUNT = 50000 
ENDING_MESSAGE = "Ok, it's late, and we have work tomorrow. Let's stop it here and continue next weekend?"

#summarizer = SummarizationAgent()

def initialize_agents():
    # Delete history files before creating agents
    gm_history_path = "gm_history.pkl"
    player_history_path = "player_history.pkl"
    
    if os.path.exists(gm_history_path):
        os.remove(gm_history_path)
    
    if os.path.exists(player_history_path):
        os.remove(player_history_path)


    with open("gm_header.txt") as f:
        gm_header = all_header + f.read()
        gm = Agent("GM", gm_header, gm_history_path)

    with open("player_header.txt") as f:
        player_header = all_header + f.read()
        player = Agent("Player", player_header, player_history_path)

    return gm, player



def run_conversation(session_word_limit):
    message = "What is your character's name, class, and what do they look like?"
    current_agent = player
    session_word_count = 0

    try:
        while session_word_count < session_word_limit:
            print(f"{current_agent.name}:")
            print(f"TOTAL WORD COUNT {sys.stdout.word_count}")
            print(f"SESSION WORD COUNT{session_word_count}")
            resp = current_agent.message(message)
            message = resp.content
            session_word_count += len(message.split())

            if current_agent == player:
                current_agent = gm
            else:
                current_agent = player
    except KeyboardInterrupt:
        print("Chat session interrupted.")
    except Exception as e:
        print(f"An error occurred: {e}")

SESSION_WORD_LIMIT = 4000

try:
    while sys.stdout.word_count < TARGET_WORD_COUNT - SESSION_WORD_LIMIT: # Buffer to ensure we don't exceed target count
        gm, player = initialize_agents()
        run_conversation(SESSION_WORD_LIMIT)
        
        # Add session ending:
        gm.message(ENDING_MESSAGE)
        time.sleep(2)  # Wait for 2 seconds between sessions
except KeyboardInterrupt:
    print("Entire simulation interrupted.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Stop capturing and close the file
    sys.stdout = original_stdout
    file.close()
