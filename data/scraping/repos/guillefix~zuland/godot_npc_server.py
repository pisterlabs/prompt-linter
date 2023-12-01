import openai
import os
from flask import Flask, request, jsonify
import config
from collections import defaultdict
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
# MEMORY
from langchain.schema import Document

# Define custom exceptions
class ChatGPTError(Exception):
    pass

class AIAgent:
    def __init__(self):
        self.api_key = os.environ["OPENAI_API_KEY"]
        openai.api_key = self.api_key
        self.short_term_memory = 3 # number of recent interactions to keep in short term memory
        self.max_retrieved_memories = 3
        self.memory_initialized = False
        self.conversation = []
        self.prompt = """
            # Introduction

            You are acting as an agent living in a simulated 2 dimensional universe. Your goal is to exist as best as you see fit and meet your needs.

            # Capabilities

            You have a limited set of capabilities. They are listed below:

            * Wait
            * walkTo (to either npcs or buildings)
            * talkTo (Make sure you only talk to those who are nearby)

            # Responses

            You must supply your responses in the form of valid JSON objects.  Your responses will specify which of the above actions you intend to take.  The following is an example of a valid response:

            *type: Type of action to take.  Valid values are: move, wait, walkTo, talkTo
            *where: Only for type "walkTo" , "talkTo". Valid values are(dont respond with any other value): dome, lighthouse, ethereum_state1, ethereum_state2, honky_ai, meow_meow_singer, longevity_vampire , network_fisherman, vvzalik
            *thought: For all types, the reason why you are doing this action
            *talking: Only for type "talkTo", what you are saying to the other person
            *relationship: Only for type "talkTo", what is your relationship with the other person
            *feeling: For all types, an emoji to represent your feeling

            {{
            "action": {{
                "type": "walkTo",
                "where": "dome",
                "thought": "Hello World",
                "talking: "Hello, How are you?",
                "relationship": "Friendly",
                "feeling": "❤️"
            }}
            }}

            # Perceptions

            You will have access to data to help you make your decisions on what to do next.

            For now, this is the information you have access to:

            What is my name?
            {npc_name}

            Who am I?
            {npc_desc}

            Location:
            {location}

            Activity:
            {activity}
            activity = data.get('activity', '')

            Nearby Players:
            {nearby_players}

            Inventory:
            {inventory}

            Message:
            {message}

            Funds:
            {funds}

            Give JSON response only and nothing more, indicating the next move. REMEMBER, ONLY RESPOND WITH VALID JSON
        """

    def get_memories(self, query):
        if not self.memory_initialized:
            return []
        docs = self.memory_retriever.get_relevant_documents(query)
        memories = []
        for d in docs:
            i = d.metadata["index"]
            mem = self.conversation[i]
            if mem["role"] == "user":
                memories.append(mem)
                memories.append(self.conversation[i+1])
            elif mem["role"] == "assistant":
                memories.append(self.conversation[i-1])
                memories.append(mem)

        return memories[:self.max_retrieved_memories]

    def add_memories(self, docs):
        if not self.memory_initialized:
            self.vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
            # vector_store.add_texts(["awawawa"])
            self.memory_retriever = self.vector_store.as_retriever()
            self.memory_initialized = True
        else:
            self.vector_store.add_documents(docs)

    def interact(self, data):
        new_prompt = self.prompt.format(**data)
        #Short term memory would be self.conversation (last 5) , Add long term memory
        self.conversation.append({"role": "user", "content": new_prompt})
        if self.memory_initialized:
            #memories = self.get_memories(new_prompt)
            # print("awoo")
            memories = []
        else:
            memories = []
        print("len(memories)", len(memories))
        response = self.chatgpt_with_retry(self.conversation[-self.short_term_memory*2:], memories)
        self.conversation.append({"role": "assistant", "content": response})
        # N = len(self.conversation)
        # if new_prompt is not None and response is not None:
        #     self.add_memories([
        #         Document(page_content=self.conversation[-1]["content"], metadata={"index": N-1}),
        #         Document(page_content=self.conversation[-2]["content"], metadata={"index": N-2})
        #     ])
        return response

    def chatgpt(self, conversation, memories, temperature=0.75, frequency_penalty=0.2, presence_penalty=0):
        messages_input = conversation.copy()
        prompt = [{"role": "system", "content": self.prompt}]
        messages_input = memories + messages_input
        messages_input.insert(0, prompt[0])

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            messages=messages_input)
        chat_response = completion['choices'][0]['message']['content']
        return chat_response

    def chatgpt_with_retry(self, conversation, memories, temperature=0.75, frequency_penalty=0.2, presence_penalty=0, retries=3):
        for i in range(retries):
            try:
                return self.chatgpt(conversation, memories, temperature, frequency_penalty, presence_penalty)
            except Exception as e:
                if i < retries - 1:
                    print(f"Error in chatgpt attempt {i + 1}: {e}. Retrying...")
                else:
                    print(f"Error in chatgpt attempt {i + 1}: {e}. No more retries.")
        return None


app = Flask(__name__)
# chatbot = ChatGPT(config.OPENAI_API_KEY)
chatbots = defaultdict(lambda: AIAgent()) # This was changed


@app.route('/chat', methods=['POST'])
def get_chat_response():
    data = request.get_json()

    # populate with defaults if data missing
    data["npc_name"] = data.get('npc_name', '')
    data["npc_desc"] = data.get('npc_desc', '')
    data["location"] = data.get('location', '')
    data["activity"] = data.get('activity', '')
    data["nearby_players"] = data.get('nearby_players', [])
    data["inventory"] = data.get('inventory', [])
    data["message"] = data.get('message', '')
    data["funds"] = data.get('funds', 0)

    npc = data.get('npc', '')

    chatbot = chatbots[npc]

    # print("Chat bot Memory", chatbot.conversation)

    response = chatbot.interact(data)
    # action = extract_action_from_response(response)  # You'd need to implement this function
    #print("Response: "+ response)
    print(response)
    return jsonify(response)

import base64
import os
import requests

engine_id = "stable-diffusion-v1-5"
api_host = os.getenv('API_HOST', 'https://api.stability.ai')
api_key = config.STABILITY_API_KEY

if api_key is None:
    raise Exception("Missing Stability API key.")

import uuid

@app.route('/t2i', methods=['POST'])
def generate_image():
    data = request.get_json()

    # populate with defaults if data missing
    prompt = data.get('prompt', 'A cool image')
    if prompt == "":
        prompt = "A cool image"

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": prompt
                }
            ],
            "cfg_scale": 7,
            "clip_guidance_preset": "FAST_BLUE",
            "height": 512,
            "width": 512,
            "samples": 1,
            "steps": 15,
        },
    )
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    # for i, image in enumerate(data["artifacts"]):
    image = data["artifacts"][0]
    filename = f"v1_txt2img_{str(uuid.uuid1())}.png"
    path = f"./../2.GODOT4-RPG/generated/" + filename
    with open(path, "wb") as f:
        f.write(base64.b64decode(image["base64"]))

    response = {"generated_img":filename}
    print(response)
    return response



if __name__ == '__main__':
    app.run(port=5000, debug=True)
