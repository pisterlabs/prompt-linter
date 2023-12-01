import openai
import threading

openai.api_key = "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"

def pretty_print_dialogue(dialogue):
    for i, item in enumerate(dialogue):
        # Handle non-dict items with a JSON-like string provided
        if isinstance(item, dict):
            print(f"{item['role']}: ", end="")
            print(f"{item['content'].strip()}")  # strip() to remove leading/trailing whitespaces
        #print("" * 30)  # Separator between turns


class PlayerGPT:

    api_data = {
        'response': None,
        'status': "used",
        'lock': threading.Lock()
    }

    def make_api_call(self):
        # Your API call code here...
        with self.api_data['lock']:
            self.api_data['status'] = "waiting"
        
        # Include only the current state:
        payload_simple = [self.system_message,
             self.history[-1]]
        # Include all chat history:
        payload_history = self.history

        # Include just some of the most recent history plus the first message:
        if len(self.history) > 3:
            payload_recent = [self.history[0]] + self.history[-3:]
        else:
            payload_recent = self.history

        # Models:
        model_gpt4 = "gpt-4"
        model_gpt3 = "gpt-3.5-turbo"

        model = model_gpt3
        payload = payload_recent
        print(f"ChatCompletion call to {model}:")
        pretty_print_dialogue(payload)
        response = openai.ChatCompletion.create(
            model=model_gpt4,
            messages=payload
        )
        
        # Lock the shared data structure to safely update it
        with self.api_data['lock']:
            self.api_data['response'] = response
            self.api_data['status'] = "ready"



    def __init__(self):
        self.system_message = {"role": "system", "content": """You are a terse video game player. You always say the correct key to press in response to the game."""}
        self.history = [
            self.system_message
        ]

    def get_direction(self, gamestate, note=""):
        #time.sleep(10) # Delay so we don't burn tokens so fast xD
        response = None
        with self.api_data['lock']:
            if self.api_data['status'] == "waiting":
                print(".",end="")
                return None
            elif self.api_data['status'] == "ready":
                print("!",end="\n")
                response = self.api_data['response']
                self.api_data['response'] = None
                self.api_data['status'] = "used"
            elif self.api_data['status'] == "used":
                print("+",end="")
                self.history.append({"role": "user", "content": f"What arrow key should you press?{gamestate}"})
                threading.Thread(target=self.make_api_call).start()
                return None

        self.history.append(response["choices"][0]["message"])
        reply = response["choices"][0]["message"]["content"]

        #print("==="*10)
        #print(gamestate)
        print(" - "*10)
        print(reply)
        print("==="*10)
        print("") # new line
        directions = ["left", "right", "up", "down"]
        for direction in directions:
            if direction in reply.lower():
                return direction

        # There was no direction in the reply, so we need to reprompt the model with a note
        return self.get_direction(gamestate, note="Sorry, your previous answer did not include an arrow key direction (left, right, up, down).")

