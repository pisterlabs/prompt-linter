
from langchain.schema import messages_to_dict
import json 
import os
from utils.config import SHORT_TERM_CHAT_HISTORY



def load_short_term_memory_from_json():
    # Check if there exits a chat history
    if os.path.exists(SHORT_TERM_CHAT_HISTORY):
        with open(SHORT_TERM_CHAT_HISTORY, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    # Return empty list if there has not been anything yet 
    return []


def save_short_term_memory_to_json(short_term_memory, moving_summary_buffer):
    try:
        # Short-term memory is most recent chat record
        message_dict = messages_to_dict(short_term_memory)
        print("###Converted short term chat history to dict!###")
        with open(SHORT_TERM_CHAT_HISTORY, "w", encoding="utf-8") as f:
            data = {
                "short_term_chat_history": message_dict, 
                "moving_summary_buffer": moving_summary_buffer 
            }
            json.dump(data, f, indent=4)
        print("###Short-term chat hisotry saved to json successfully!###")
        print()
    except Exception as e:
        print("!!!Failed to saving chat history into json!!!")
        print(e)
        print()    


def delete_short_term_history():
    with open(SHORT_TERM_CHAT_HISTORY, "w", encoding="utf-8") as f:
        data = []
        json.dump(data, f)
        print("###JSON chat history successfully deleted.###")
        print()




