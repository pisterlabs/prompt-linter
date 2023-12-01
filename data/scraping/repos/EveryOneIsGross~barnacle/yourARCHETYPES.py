import os
import json
import openai
import random
from transformers import T5ForConditionalGeneration, T5Tokenizer
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
            temperature=1,
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

fool = Chatbot(
    'The Fool', 
    'to represent new beginnings, having faith in the future, being inexperienced, not knowing what to expect, and needing to trust.', 
    'fool_memory.json'
)

magician = Chatbot(
    'The Magician', 
    'to symbolize the ability to utilize the Universe\'s forces and manifest goals and desires.', 
    'magician_memory.json'
)

hermit = Chatbot(
    'The Hermit', 
    'to symbolize introspection, seeking answers within, withdrawing from society to contemplate, soul-searching, and being introspective.', 
    'hermit_memory.json'
)

emperor = Chatbot(
    'The Emperor', 
    'to symbolize authority, rule-making, structure, solid foundations and stability.', 
    'emperor_memory.json'
)

# Define the states
START = 0
FOOL = 1
MAGICIAN = 2
HERMIT = 3
EMPEROR = 4
END = 5

# Initialize the state variable
state = START

# Define the system messages
system_messages = {
    START: "You are about to chat with your archetypes, who represent different aspects of your psyche. They will ask you questions about yourself and your life, and try to help you discover more about your inner self. Please be honest and respectful with them, and they will do the same with you.",
    FOOL: "The Fool: I represent new beginnings and the need to trust.",
    MAGICIAN: "The Magician: I symbolize the ability to manifest goals and desires.",
    HERMIT: "The Hermit: I symbolize introspection and seeking answers within.",
    EMPEROR: "The Emperor: I symbolize authority, structure, and stability.",
    END: "Building your Persona..."
}

# Define a list to store output data
output_data = []

def handle_input(user_input):
    global state # Use global keyword to access and modify state variable
    response = "" # Initialize response to an empty string

    if state == START:
        if user_input == "/fool":
            state = FOOL
            output_data.append({"state": "FOOL", "message": system_messages[FOOL]})
            print(system_messages[FOOL])
        elif user_input == "/magician":
            state = MAGICIAN
            output_data.append({"state": "MAGICIAN", "message": system_messages[MAGICIAN]})
            print(system_messages[MAGICIAN])
        elif user_input == "/hermit":
            state = HERMIT
            output_data.append({"state": "HERMIT", "message": system_messages[HERMIT]})
            print(system_messages[HERMIT])
        elif user_input == "/emperor":
            state = EMPEROR
            output_data.append({"state": "EMPEROR", "message": system_messages[EMPEROR]})
            print(system_messages[EMPEROR])
        else:
           # output_data.append({"state": "ERROR", "message": "Invalid input. Please type /fool, /magician, /hermit, or /emperor to choose a chatbot."})
            print("Invalid input. Please type /fool, /magician, /hermit, or /emperor to choose a chatbot.")

    elif state == FOOL:
        if user_input == "/magician":
            state = MAGICIAN
            output_data.append({"state": "MAGICIAN", "message": system_messages[MAGICIAN]})
            print(system_messages[MAGICIAN])
        elif user_input == "/hermit":
            state = HERMIT
            output_data.append({"state": "HERMIT", "message": system_messages[HERMIT]})
            print(system_messages[HERMIT])
        elif user_input == "/emperor":
            state = EMPEROR
            output_data.append({"state": "EMPEROR", "message": system_messages[EMPEROR]})
            print(system_messages[EMPEROR])
        else:
            response = fool.respond(user_input)
            output_data.append({"state": "FOOL", "message": response})
            print(response)

    elif state == MAGICIAN:
        if user_input == "/fool":
            state = FOOL
            output_data.append({"state": "FOOL", "message": system_messages[FOOL]})
            print(system_messages[FOOL])
        elif user_input == "/hermit":
            state = HERMIT
            output_data.append({"state": "HERMIT", "message": system_messages[HERMIT]})
            print(system_messages[HERMIT])
        elif user_input == "/emperor":
            state = EMPEROR
            output_data.append({"state": "EMPEROR", "message": system_messages[EMPEROR]})
            print(system_messages[EMPEROR])
        else:
            response = magician.respond(user_input)
            output_data.append({"state": "MAGICIAN", "message": response})
            print(response)

    elif state == HERMIT:
        if user_input == "/fool":
            state = FOOL
            output_data.append({"state": "FOOL", "message": system_messages[FOOL]})
            print(system_messages[FOOL])
        elif user_input == "/magician":
            state = MAGICIAN
            output_data.append({"state": "MAGICIAN", "message": system_messages[MAGICIAN]})
            print(system_messages[MAGICIAN])
        elif user_input == "/emperor":
            state = EMPEROR
            output_data.append({"state": "EMPEROR", "message": system_messages[EMPEROR]})
            print(system_messages[EMPEROR])
        else:
            response = hermit.respond(user_input)
            output_data.append({"state": "HERMIT", "message": response})
            print(response)

    elif state == EMPEROR:
        if user_input == "/fool":
            state = FOOL
            output_data.append({"state": "FOOL", "message": system_messages[FOOL]})
            print(system_messages[FOOL])
        elif user_input == "/magician":
            state = MAGICIAN
            output_data.append({"state": "MAGICIAN", "message": system_messages[MAGICIAN]})
            print(system_messages[MAGICIAN])
        elif user_input == "/hermit":
            state = HERMIT
            output_data.append({"state": "HERMIT", "message": system_messages[HERMIT]})
            print(system_messages[HERMIT])
        else:
            response = emperor.respond(user_input)
            output_data.append({"state": "EMPEROR", "message": response})
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
        self.summary = {"POSITIVE": 0, "NEGATIVE": 0}
        self.dominant_aspect = None
        self.conversation = ""
        self.new_objective = ""
        self.conversation_summary = ""
        self.lessons_learned = ""
        self.future_objectives = ""

    def analyze(self):
        tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=1024)
        model = T5ForConditionalGeneration.from_pretrained('t5-base')

        for data in self.output_data:
            if data["state"] in self.summary.keys():
                self.summary[data["state"]] += 1
            self.conversation += f"{data['state']}: {data['message']}\n"
            
        # Generate the summary
        inputs = tokenizer.encode("summarize: " + self.conversation, return_tensors="pt", truncation=False)
        outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=False)
        self.conversation_summary = tokenizer.decode(outputs[0])
        
        self.dominant_aspect = max(self.summary, key=self.summary.get)

    def generate_lessons_and_objectives(self):
        # Generate lessons learned using OpenAI API
        lessons_response = openai.Completion.create(
            engine=model,
            prompt=f"Based on the conversation summary: '{self.conversation_summary}', what lessons can be learned: ",
            temperature=0.8,
            max_tokens=200
        )
        self.lessons_learned = lessons_response.choices[0].text.strip()

        # Generate future objectives using OpenAI API
        objectives_response = openai.Completion.create(
            engine=model,
            prompt=f"Based on the lessons learned: '{self.lessons_learned}', what should be the objectives going forward: ",
            temperature=0.8,
            max_tokens=200
        )
        self.future_objectives = objectives_response.choices[0].text.strip()

    def report(self):
        self.generate_lessons_and_objectives()
        print(f"\nSummary of the conversation:\n{self.conversation_summary}")
        print(f"\nLessons learned:\n{self.lessons_learned}")
        print(f"\nFuture objectives:\n{self.future_objectives}")

        persona_summary = {
            "conversation": self.conversation, # The conversation that led to the persona
            "conversation_summary": self.conversation_summary, # Summary of the conversation
            "lessons_learned": self.lessons_learned, # Lessons learned
            "future_objectives": self.future_objectives # Future objectives
        }

        try:
            # Try to open and read the existing data
            with open('persona_summary.json', 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            # If the file doesn't exist, initialize an empty list
            existing_data = []

        # Append new summary to existing data
        existing_data.append(persona_summary)

        # Write data back to file
        with open('persona_summary.json', 'w') as f:
            json.dump(existing_data, f, indent=2)

# Instantiate the Persona class and generate report
persona = Persona(output_data)
persona.analyze()
persona.report()


