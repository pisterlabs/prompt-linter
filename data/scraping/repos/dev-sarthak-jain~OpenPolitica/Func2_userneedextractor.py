import os
import datetime
import json
import os
from dotenv import load_dotenv
load_dotenv()

#apikey = ""
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Assuming apikey is correctly imported
os.environ['OPENAI_API_KEY'] = os.getenv("apikey")

class UserNeedsExtractor:
    def __init__(self, temperature=0.9):
        # Initialize LLM and memory
        self.llm = OpenAI(temperature=temperature)
        self.conversation_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

        # Configure prompt templates
        self.issue_extraction_template = PromptTemplate(
            input_variables=['topic'],
            template='Given the following conversation, extract general concerns that emphasize personal effects or outcomes without referencing specific policies or solutions. Conversation: {topic}'
        )
        self.type_determination_template = PromptTemplate(
            input_variables=['concern'],
            template="Classify the following concern into a category such as Infrastructure, Healthcare, Economy, etc.: {concern}"
        )

        # Configure chains
        self.issue_extraction_chain = LLMChain(llm=self.llm, prompt=self.issue_extraction_template, verbose=True, output_key='concerns', memory=self.conversation_memory)
        self.type_determination_chain = LLMChain(llm=self.llm, prompt=self.type_determination_template, verbose=True, output_key='type')

    def extract_user_needs(self, chat_transcript):
        conversation = self._format_conversation(chat_transcript)
        concerns = [c.strip() for c in self.issue_extraction_chain.run(conversation).split('.') if c]
        structured_needs = []
        for idx, concern in enumerate(concerns):
            concern_type = self.determine_concern_type(concern)
            structured_needs.append({
                "concern_id": idx + 1,
                "description": concern,
                "type": concern_type
            })
        return structured_needs

    def _format_conversation(self, chat_transcript):
        conversation = ""
        for message in chat_transcript:
            if message['role'] in ['user', 'assistant']:
                # Add a line break and the speaker's role before each message for clarity
                speaker = "User: " if message['role'] == 'user' else "Assistant: "
                conversation += speaker + message['content'] + "\n\n"
        return conversation.strip()

    def determine_concern_type(self, concern):
        return self.type_determination_chain.run(concern)

    def generate_survey(self, structured_needs, q_distribution):
        options = []
        for need in structured_needs:
            # Check if the type field contains a colon
            if ':' in need["type"]:
                # Splitting categories and individual needs
                categories = need["type"].split('\n\n')
                for category in categories:
                    if category.strip():  # Ensure the category string is not empty
                        cat_name, cat_needs_str = category.split(':')
                        cat_needs = cat_needs_str.split('\n- ')[1:]  # Skip the first empty item
                        for ind_need in cat_needs:
                            if ind_need.strip():  # Ensure the individual need is not empty
                                options.append([ind_need.strip(), 0])
            else:
                # If no colon is present, treat the whole type field as a single category
                options.append([need["type"].strip(), 0])

        survey = {
            "questions": len(options),
            "user_needs": len(options),
            "options": options
        }
        return survey
   
    def save_concerns_to_database(self, structured_needs):
        # Implement database integration here
        pass

    def run(self, chat_transcript):
        structured_needs = self.extract_user_needs(chat_transcript)
        survey = self.generate_survey(structured_needs, [-3, -2, -2, -1, -1, -1, 0, 0, 1, 1, 1, 2, 2, 3])
        self.save_concerns_to_database(structured_needs)
        return structured_needs, survey
'''
# Usage Example
extractor = UserNeedsExtractor()
chat_transcript = [
    {"role": "user", "content": "Hi, I'm really concerned about the lack of public parks in our area."},
    {"role": "assistant", "content": "That's a valid concern. Do you think having more public parks would benefit the community?"},
    {"role": "user", "content": "Absolutely, especially for families and children. It's important to have green spaces."},
    {"role": "assistant", "content": "I understand. Green spaces are essential for a healthy community. Anything else on your mind?"},
    {"role": "user", "content": "Yes, I'm also worried about the public transport system. It's not very reliable."},
    {"role": "assistant", "content": "Reliability of public transportation is indeed crucial for daily commuters. How often do you face issues with it?"},
    {"role": "user", "content": "Almost every day. The buses are either late or too crowded."},
    {"role": "assistant", "content": "That must be frustrating. Reliable and comfortable public transport is important. Any other issues?"},
    {"role": "user", "content": "Well, there's one more thing. I'm not very satisfied with the local health services."},
    {"role": "assistant", "content": "Can you tell me more about what's lacking in the local health services?"},
    {"role": "user", "content": "The waiting times are too long and there aren't enough specialized medical facilities."},
    {"role": "assistant", "content": "Long waiting times and lack of specialized facilities can indeed be a concern. Thank you for sharing your thoughts."}
]

data = [
    {'role': 'user', 'content': "Hi, I'm really concerned about the lack of public parks in our area."},
    {'role': 'assistant', 'content': 'System: You can contact your local government and ask them about any initiatives they have to create or improve public parks. You can also get involved in local advocacy groups that are working to increase access to green spaces in your area. Additionally, you can volunteer with local organizations that are working to improve public parks and green spaces.'},
    {'role': 'user', 'content': "especially for families and children. It's important to have green spaces."},
    {'role': 'assistant', 'content': 'System: Absolutely, green spaces are essential for providing a safe and healthy environment for families and children. You can contact your local government to ask if they are developing any projects to create or maintain these green spaces. You can also research any nearby organizations or initiatives that are working to protect and improve green spaces, and see if you can volunteer or get involved in their efforts. Additionally, you can reach out to your local schools and see if they have any initiatives for improving public parks and green spaces.'},
    {'role': 'user', 'content': "I'm also worried about the public transport system. It's not very reliable"},
    {'role': 'assistant', 'content': 'System: Public transport is an important resource for many people, and it is important to ensure that it is reliable and accessible. You can contact your local government to see what initiatives they have in place to improve public transport. Additionally, you can get involved in local advocacy groups that are working to ensure that public transport is accessible and reliable for all. You can also research any local initiatives that are working to improve public transport and see if you can volunteer to help with their efforts.'},
    {'role': 'user', 'content': 'Almost every day. The buses are either late or too crowded.'},
    {'role': 'assistant', 'content': 'System: That is definitely a problem. You can contact your local government and ask them about any initiatives they have in place to improve public transport. Additionally, you can research any local initiatives or advocacy groups that are working to ensure that public transport is reliable and accessible. You can also contact your local bus company to see if they have any initiatives or plans to improve their service.'},
    {'role': 'user', 'content': "there's one more thing. I'm not very satisfied with the local health services."},
    {'role': 'assistant', 'content': 'System: Access to quality health services is essential for the well-being of all citizens. You can contact your local government and ask them about any initiatives they have to improve the quality of local health services. Additionally, you can research any local initiatives or advocacy groups that are working to ensure that health services are accessible and reliable for all. You can also contact your local health services and see if they have any initiatives or plans to improve their services.'},
    {'role': 'user', 'content': "The waiting times are too long and there aren't enough specialized medical facilities."},
    {'role': 'assistant', 'content': 'System: That is definitely a problem. You can contact your local government and ask them about any initiatives they have in place to improve the quality of local health services, such as reducing waiting times and increasing access to specialist medical facilities. Additionally, you can research any local initiatives or advocacy groups that are working to ensure that health services are accessible and reliable for all. You can also contact your local health services and see if they have any initiatives or plans to improve their services.'}
]


structured_needs, survey = extractor.run(chat_transcript)
print(structured_needs)
print(survey)
'''
# User Needs Extraction Output:
# The output is a list of dictionaries, where each dictionary represents an extracted user need from the conversation. In your provided example, there is one extracted need:

# [
#     {
#         "concern_id": 1,
#         "description": "General concerns: \n- Lack of green spaces \n- Reliability and comfort of public transport\n- Long waiting times and lack of specialized medical facilities",
#         "type": "\n\nInfrastructure"
#     }
# ]



# User Needs Extraction Output:
# The output is a list of dictionaries, where each dictionary represents an extracted user need from the conversation. In your provided example, there is one extracted need:
# json
# Copy code
# [
#     {
#         "concern_id": 1,
#         "description": "General concerns: \n- Lack of green spaces \n- Reliability and comfort of public transport\n- Long waiting times and lack of specialized medical facilities",
#         "type": "\n\nInfrastructure"
#     }
# ]
# Here's what each key-value pair represents:
# concern_id: A unique identifier for the concern (in this case, 1).
# description: A text description of the concern. This includes a series of sub-concerns, each on a new line, prefixed with a dash (-).
# type: The classified category of the concern. Here, it's categorized as Infrastructure, but the type starts with new line characters which might be an artifact from the text generation process.

# Survey Output:
# The survey output is a dictionary where each key represents a question related to a user need. The value is another dictionary containing the question text and the possible answers (rankings based on the Q methodology).

# {
#     "Question1": {
#         "question": "How would you rank the issue: 'General concerns: \n- Lack of green spaces \n- Reliability and comfort of public transport\n- Long waiting times and lack of specialized medical facilities'?",
#         "answers": [-3, -2, -2, -1, -1, -1, 0, 0, 1, 1, 1, 2, 2, 3]
#     }
# }

# In this format:
# Question1 is the key for the first question.
# Inside, there are two keys: question (the text of the question) and answers (an array of integers representing the Q methodology distribution for ranking the concern).
# The question text includes the description of the user need. The formatting, such as newline characters (\n), helps to separate different aspects of the general concern, making it easier for respondents to understand and rank.

# Integration Guide for Front-End Developers: User Needs Extraction and Survey Generation
# Overview

# The User Needs Extraction and Survey Generation feature is designed to analyze chat transcripts, extract key user concerns (user needs), and generate a tailored survey based on these concerns. This document provides guidance for front-end developers on integrating this feature into a web application or platform.

# Key Components

# UserNeedsExtractor Class: This is the core backend class that processes chat transcripts to extract user needs and generate surveys.
# Input: A chat transcript formatted as a list of dictionaries, with each entry containing a "role" (either "user" or "assistant") and a "message".
# Output:
# Structured Needs: A list of dictionaries, each representing a user need extracted from the chat. Each dictionary includes a unique concern_id, a description of the concern, and its type (e.g., Infrastructure, Healthcare).
# Survey: A list of survey questions, each related to a specific user need. The survey questions are designed to gauge the user's prioritization of these needs.
# Integration Steps

# Chat Transcript Collection:
# Ensure that the chat interface collects and formats user and assistant messages in the required list of dictionaries format.
# Extracting User Needs:
# On completion of a chat session, send the transcript to the backend UserNeedsExtractor class.
# The backend will process the transcript and extract key user needs.
# Survey Generation:
# The backend generates a survey based on the extracted user needs.
# Each survey question asks the user to rank a specific need based on its importance to them.
# Displaying the Survey:
# The front end should dynamically render the survey questions.
# Consider using a slider or a similar input mechanism for users to rank each concern.
# Ensure each question is clearly associated with its corresponding user need.
# Collecting Survey Responses:
# Collect user responses and map them back to the corresponding user needs.
# Store these responses in a structured format that includes both the user need and its ranking.
# Sending Responses to the Backend:
# Once the user completes the survey, send the structured responses back to the backend.
# The backend can then use these rankings for further analysis or policy generation.

# {
#   "Question1": {
#     "question": "How important is the issue: 'General concerns: - Lack of green spaces - Reliability and comfort of public transport - Long waiting times and lack of specialized medical facilities'?",
#     "answers": [-3, -2, -2, -1, -1, -1, 0, 0, 1, 1, 1, 2, 2, 3]
#   }
# }
# Question1: The key for the first survey question.
# question: The text of the survey question, directly related to a user need.
# answers: The range of rankings the user can select from, indicating the importance of the issue.