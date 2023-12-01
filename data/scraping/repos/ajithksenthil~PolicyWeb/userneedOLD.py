import os
import datetime
import json

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Assuming apikey is correctly imported
os.environ['OPENAI_API_KEY'] = apikey

class UserNeedsExtractor:
    def __init__(self, temperature=0.9):
        # Initialize LLM and memory
        self.llm = OpenAI(temperature=temperature)
        self.conversation_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

        # Configure prompt templates
        self.issue_extraction_template = PromptTemplate(
            input_variables=['message'],
            template='Analyze the following message to extract general concerns that emphasize personal effects or outcomes without referencing specific policies or solutions: {message}'
        )
        self.type_determination_template = PromptTemplate(
            input_variables=['concern'],
            template="Classify the following concern into a category such as Infrastructure, Healthcare, Economy, etc.: {concern}"
        )

        # Configure chains
        self.issue_extraction_chain = LLMChain(llm=self.llm, prompt=self.issue_extraction_template, verbose=True, output_key='concerns', memory=self.conversation_memory)
        self.type_determination_chain = LLMChain(llm=self.llm, prompt=self.type_determination_template, verbose=True, output_key='type')

    def extract_user_needs(self, chat_transcript):
        structured_needs = []
        for message in chat_transcript:
            if message['role'] == 'user':
                extracted_concerns = self.issue_extraction_chain.run(message['content']).split('.')
                for idx, concern in enumerate(extracted_concerns):
                    concern = concern.strip()
                    if concern:
                        concern_type = self.determine_concern_type(concern)
                        structured_needs.append({
                            "concern_id": f"{message['role']}_msg_{idx + 1}",
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
        survey = []
        for need in structured_needs:
            survey.append({
                "question": f"How important is the issue: '{need['description']}' to you?",
                "options": q_distribution
            })
        return survey

    def save_concerns_to_database(self, structured_needs):
        # Implement database integration here
        pass

    def run(self, chat_transcript):
        structured_needs = self.extract_user_needs(chat_transcript)
        survey = self.generate_survey(structured_needs, [-3, -2, -2, -1, -1, -1, 0, 0, 1, 1, 1, 2, 2, 3])
        self.save_concerns_to_database(structured_needs)
        return structured_needs, survey

# Usage Example
extractor = UserNeedsExtractor()
chat_transcript = [
    {"role": "user", "content": "I'm worried about the increasing traffic in the city."},
    {"role": "assistant", "content": "That sounds concerning. Can you tell me more about how it affects you?"}
    # ... rest of the chat transcript
]
structured_needs, survey = extractor.run(chat_transcript)
