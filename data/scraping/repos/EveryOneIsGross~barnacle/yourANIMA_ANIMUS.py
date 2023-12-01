import os
import json
import openai
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Assign OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv('OPENAI_ENGINE')

class Chatbot:
    def __init__(self, name, mission, memory_file):
        self.name = name
        self.mission = mission
        self.memory_file = memory_file
        # Load memory from file
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                self.memory = json.load(f)
        else:
            self.memory = []

    def save_memory(self):
        # Save memory to file
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def respond(self, message):
        # Load memories
        past_memories = ""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                memory_list = json.load(f)
                # Check if there are at least 2 memories
                if len(memory_list) >= 2:
                    # Randomly select 2 memories
                    selected_memories = random.sample(memory_list, 1)
                    for mem in selected_memories:
                        past_memories += f"{self.name}: {mem['response']}\nUser: {mem['question']}\n"

        # Generate response with past memories included in the prompt
        response = openai.Completion.create(
            engine=model,
            # Change the prompt to include past memories, the chatbot's name, mission, and the user's message
            prompt=f"Considering our previous thoughts{past_memories}, The {self.name} chatbot, whose mission is {self.mission}, received a message: '{message}'.\n{self.name}: {message}\nUser: ",
            temperature=0.9,
            max_tokens=300
        )
        response_text = response.choices[0].text.strip()

        # Update memory
        self.memory.append({
            "name": self.name,
            "question": message,
            "response": response_text
        })

        # Save memory
        self.save_memory()

        return response_text

anima = Chatbot(
    'Anima', 
    'to nurture emotional understanding, foster creativity, and hone intuitive abilities. '
    'You are associated with traditionally feminine qualities: encourages exploration '
    'of feelings and dreams, and supports self-discovery and emotional growth.', 
    'anima_memory.json'
)

animus = Chatbot(
    'Animus', 
    'to strengthen decision-making skills, cultivate assertiveness, and develop logical thinking. '
    'You are embodied traditionally masculine traits: aims to help individuals overcome '
    'challenges and discover their purpose.', 
    'animus_memory.json'
)

# Define the states
START = 0
ANIMA = 1
ANIMUS = 2
END = 3

# Initialize the state variable
state = START

# Define the system messages
system_messages = {
    START: "You are about to chat with your Anima and Animus, who represent the feminine and masculine aspects of your psyche. They will ask you questions about yourself and your life, and try to help you discover more about your inner self. Please be honest and respectful with them, and they will do the same with you.",
    ANIMA: "Anima: I advocate for the feminine aspect of your personality.",
    ANIMUS: "Animus: I advocate for the masculine aspect of your personality.",
    END: "Building your Persona..."
}

# Define a list to store output data
output_data = []

# Define a function to handle user input
def handle_input(user_input):
    global state # Use global keyword to access and modify state variable
    response = "" # Initialize response to an empty string

    if state == START:
        if user_input == "/anima":
            state = ANIMA
            output_data.append({"state": "ANIMA", "message": system_messages[ANIMA]})
            print(system_messages[ANIMA])
        elif user_input == "/animus":
            state = ANIMUS
            output_data.append({"state": "ANIMUS", "message": system_messages[ANIMUS]})
            print(system_messages[ANIMUS])
        else:
            output_data.append({"state": "ERROR", "message": "Invalid input. Please type /anima or /animus to choose a chatbot."})
            print("Invalid input. Please type /anima or /animus to choose a chatbot.")

    elif state == ANIMA:
        if user_input == "/animus":
            state = ANIMUS
            output_data.append({"state": "ANIMUS", "message": system_messages[ANIMUS]})
            print(system_messages[ANIMUS])
        else:
            response = anima.respond(user_input)
            output_data.append({"state": "ANIMA", "message": response})
            print(response)

    elif state == ANIMUS:
        if user_input == "/anima":
            state = ANIMA
            output_data.append({"state": "ANIMA", "message": system_messages[ANIMA]})
            print(system_messages[ANIMA])
        else:
            response = animus.respond(user_input)
            output_data.append({"state": "ANIMUS", "message": response})
            print(response)

    elif state == END:
        pass


# Start the conversation by printing the system message for START state
print(system_messages[START])

# Loop until the user types /end
while True:
    # Get user input
    user_input = input("User says: ")
    # Check if user input is /end (ending conversation)
    if user_input == "/end":
        # Change state to END
        state = END
        # Print system message for END state
        print(system_messages[END])
        # Break the loop
        break
    else:
        # Handle user input based on current state
        handle_input(user_input)

# At the end of your script, write output data to a JSON file
with open('output.json', 'w') as f:
    json.dump(output_data, f, indent=2)

class Persona:
    def __init__(self, output_data):
        self.output_data = output_data
        self.summary = {"ANIMA": 0, "ANIMUS": 0}
        self.dominant_aspect = None
        self.conversation = ""
        self.new_objective = ""
        self.lessons_learned = ""  # New attribute for lessons learned
        self.future_objectives = ""  # New attribute for future objectives

    def analyze(self):
        for data in self.output_data:
            if data["state"] in self.summary.keys():
                self.summary[data["state"]] += 1
                self.conversation += data["message"] + " "
        
        self.dominant_aspect = max(self.summary, key=self.summary.get)

    def generate_persona(self):
        # Generate persona using OpenAI API
        response = openai.Completion.create(
            engine=model,
            prompt=f"Persona summary: '{self.conversation}'. Persona's new objective: ",
            temperature=0.8,
            max_tokens=400
        )
        self.new_objective = response.choices[0].text.strip()


        # summary of lessons anf objectives aren't being captured from the conversation 
        # 
        # Generate lessons learned and future objectives
        response = openai.Completion.create(
            engine=model,
            prompt=f"From the conversation {self.conversation}, the persona learned: ",
            temperature=0.8,
            max_tokens=200
        )
        self.lessons_learned = response.choices[0].text.strip()

        response = openai.Completion.create(
            engine=model,
            prompt=f"Based on the lessons learned {self.lessons_learned}, the persona's future objectives are: ",
            temperature=0.6,
            max_tokens=200
        )
        self.future_objectives = response.choices[0].text.strip()

    def report(self):
        self.analyze()
        self.generate_persona()
        print(f"Dominant aspect: {self.dominant_aspect}")
        print(f"Persona's new outlook: {self.new_objective}")
        print(f"Lessons learned: {self.lessons_learned}")
        print(f"Future objectives: {self.future_objectives}")

        persona_summary = {
            "summary": self.summary,
            "dominant_aspect": self.dominant_aspect,
            "new_objective": self.new_objective,
            "lessons_learned": self.lessons_learned,
            "future_objectives": self.future_objectives
        }

        try:
            # Try to open and read the existing data
            with open('persona_summary.json', 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            # If the file doesn't exist, initialize an empty list
            existing_data = []

        # If existing_data is a dictionary, convert it to a list
        if isinstance(existing_data, dict):
            existing_data = [existing_data]

        # Append new summary to existing data
        existing_data.append(persona_summary)

        # Write data back to file
        with open('persona_summary.json', 'w') as f:
            json.dump(existing_data, f, indent=2)




# Instantiate the Persona class and generate report
persona = Persona(output_data)
persona.report()
