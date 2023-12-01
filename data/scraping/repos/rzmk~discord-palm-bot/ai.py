# Standard imports
import json
from os import getenv, path

# Third-party imports
from langchain.chat_models import ChatGooglePalm
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Local imports
from .config import get_config

# Config
PALM_API_KEY = getenv("PALM_API_KEY")

# Set up the chat model.
chat_model = ChatGooglePalm(google_api_key=PALM_API_KEY)


async def write_to_jsonl(author: str, content: str):
    # Add prompt to message history in messages.jsonl.
    with open("messages.jsonl", "a") as f:
        entry = {"author": author, "content": content}
        json.dump(entry, f)
        f.write("\n")


async def get_clean_prompt(client, message, name) -> str:
    # Get the message as the prompt and the user's display name as its name.
    clean_prompt: str = message.content
    prefix = await get_config("prefix")
    if message.content.startswith(prefix):
        clean_prompt = clean_prompt[len(prefix) :]
    if client.user.mentioned_in(message):
        clean_prompt = clean_prompt.replace(f"<@{client.user.id}>", f"@{name}")
    return clean_prompt


class ChatOutputGenerator:
    def __init__(self):
        self.messages = []
        # Create messages.jsonl if it doesn't exist.
        if not path.exists("messages.jsonl"):
            with open("messages.jsonl", "w") as f:
                f.write("")
        with open("messages.jsonl", "r") as file:
            for line in file:
                message_dict: dict = json.loads(line)
                if message_dict["author"] == "user":
                    self.messages.append(HumanMessage(content=message_dict["content"]))
                elif message_dict["author"] == "bot":
                    self.messages.append(AIMessage(content=message_dict["content"]))

    async def generate_chat_output(self, client, message, name):
        prompt = await get_clean_prompt(client, message, name)
        # Generate chat response.
        try:
            # If self.messages is greater than 20000 bytes, remove the first messages until it is less than 20000 bytes.
            messages = map(lambda message: message.content, self.messages)
            while len(json.dumps(list(messages))) > 20000:
                self.messages.pop(0)
            # Prepend the system message context to the message history.
            if len(self.messages) == 0 or self.messages[0] is not SystemMessage:
                system_message_content = f"Be a helpful Discord bot named '{name}', and you will be referred to as '@{name}'."
                system_message = SystemMessage(content=system_message_content)
                self.messages.insert(0, system_message)
            # Generate the AI's chat response.
            response = chat_model(self.messages + [HumanMessage(content=prompt)])
            ai_content = response.content if response.content else "..."
            # If the last message was from the user, add an ellipsis to the message history.
            if len(self.messages) > 0 and self.messages[-1] is HumanMessage:
                self.messages.append(AIMessage(content="..."))
                await write_to_jsonl("bot", "...")
            # Add prompt to message history.
            self.messages.append(HumanMessage(content=prompt))
            await write_to_jsonl("user", prompt)
            # Add response to message history.
            self.messages.append(AIMessage(content=ai_content))
            await write_to_jsonl("bot", ai_content)
            return ai_content
        except Exception as e:
            print("Error generating response:")
            print(e)
            # If the last message was from the user, delete it from the message history.
            if len(self.messages) > 0 and self.messages[-1] is HumanMessage:
                self.messages.pop()
            return None
