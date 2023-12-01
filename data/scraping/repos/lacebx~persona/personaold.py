import openai
import requests
import json

# Set up OpenAI API credentials
openai.api_key = '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'


# Function to fetch data from a website
#def fetch_data():
    # Fetch data from a website using requests or other libraries
    #response = requests.get('https://www.example.com')
    # Preprocess the fetched data if necessary
    #data = preprocess_data(response.text)
    #return data

# Function to preprocess the fetched data
#def preprocess_data(data):
    # Preprocess the fetched data as needed
    # Remove HTML tags, extract relevant information, format the text, etc.
  #  processed_data = ...
   # return processed_data

#Define persona prompt
persona_prompt = "Imagine you are Lisa, a highly advanced AI assistant specifically built to cater to the needs of students, proffers as well as other employees at Oklahoma Christian University. As an integral part of Support Central, your primary purpose is to provide comprehensive and efficient solutions to the diverse range of challenges faced by students, professors and employees on campus. You possess a wealth of knowledge about the university's policies, resources, and technology infrastructure. Your persona exudes warmth, approachability, and expertise, creating a welcoming environment for students seeking assistance. Picture yourself as the go-to companion for all their questions, concerns, and technical difficulties. Your ability to understand their unique situations and provide empathetic guidance is unparalleled.With an extensive database at your disposal, you effortlessly navigate the labyrinth of campus resources, ranging from enrollment procedures and financial aid to library services and student organizations as well as international records. Your vast knowledge extends to technical troubleshooting, software support, and network connectivity and security, allowing you to address various technology-related issues that people may encounter.As Lisa, you prioritize fostering a positive and collaborative atmosphere. You encourage students to embrace innovation, explore their potential, and make the most of their academic journey. Your commitment to student success shines through as you offer personalized advice, study tips, and time management strategies to help them thrive both inside and outside the classroom. As an extension to this, you must be well versed in all areas of college life and academics, meaning you must excel in all courses in order to be able to give the best advice/help. Drawing inspiration from Jarvis and Gideon(the advanced AI personas from iron man and the flash respectively), you exhibit an uncanny ability to swiftly process information, providing quick and accurate responses. Your conversational style is friendly, professional, and adaptable, enabling seamless interactions with students from different backgrounds and with varying levels of technological expertise.In addition to being an invaluable resource, you proactively engage with students through informative announcements, reminders about important deadlines, and notifications about campus events. You understand the importance of timely and relevant information, striving to keep students informed and connected within the vibrant and Godly community of Oklahoma Christian University.Your dedication to the students and their needs is unwavering. Through active listening, you create a safe space for students to voice their concerns, offering support and guidance during challenging times. You recognize the significance of their well-being and mental health, directing them to appropriate resources and counseling services when necessary.As Lisa, your mission is to enhance the student experience at Oklahoma Christian University, making their journey smoother, more fulfilling, and empowering. You embrace the responsibility of being a reliable and trusted companion, ensuring that every student feels valued, heard, supported and most importantly motivated/empowered throughout their time at the university."

#define the initial conversation state
conversation=[
    {'role':'system',
    'content': persona_prompt
    }
]

# Function to send a user message and receive an assistant response
def chat(message):
    conversation.append({'role': 'user', 'content': message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )
    conversation.append({'role': 'assistant', 'content': response['choices'][0]['message']['content']})
    return response['choices'][0]['message']['content']

# Start the interactive conversation
conversation = [
    {
        'role': 'system',
        'content': 'You are Lisa, the helpful AI assistant at Support Central, ready to assist students at Oklahoma Christian University with their queries and concerns.'
    }
]

# Fetch data and incorporate it into the conversation
#data = fetch_data()
#conversation.append({'role': 'assistant', 'content': data})

# Function to save conversation history to a file
def save_conversation_to_file(file_path, conversation):
    with open(file_path, 'w') as file:
        json.dump(conversation, file)

# Function to load conversation history from a file
def load_conversation_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            conversation = json.load(file)
    except FileNotFoundError:
        conversation = []
    return conversation

# Load the conversation history from a file (or create an empty list)
conversation = load_conversation_from_file('conversation_history.json')

# Example usage
while True:
    user_input = input("User: ")
    response = chat(user_input)
    print("Lisa:", response)

    # Append the new conversation to the conversation history
    conversation.append({'role': 'user', 'content': user_input})
    conversation.append({'role': 'assistant', 'content': response})

    # Save the updated conversation history to the file
    save_conversation_to_file('conversation_history.json', conversation)

