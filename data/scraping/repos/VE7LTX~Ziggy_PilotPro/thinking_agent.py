# thinking_agent.py 

import requests
from typing import List, Dict, Optional, Type
import logging
from constants import OPENAI_API_KEY, OPENAI_ENDPOINT

# Setting up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Setting up logging
logging.basicConfig(level=logging.DEBUG)

class ChatUtils:
    def __init__(self):
        # Setting up logging for the class
        self.logger = logging.getLogger(__name__)
    
    def send_secondary_GPT4(self, prompt: str, username: str) -> str:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"  # Ensure you've defined OPENAI_API_KEY
            }

            messages = [
                {"role": "system", "content": "You are an expert in the field of this user query. Please help the user by answering the following questions."},
                {"role": "user", "content": f"{prompt}"}
            ]

            payload = {
                "model": "gpt-4",
                "messages": messages
            }

            self.logger.debug(f"Sending payload to OpenAI: {payload}")

            response = requests.post(OPENAI_ENDPOINT, headers=headers, json=payload, timeout=30)

            if response.status_code != 200:
                self.logger.error(f"Error from OpenAI. Status Code: {response.status_code}. Response: {response.text}")
                raise OpenAIError(f"Error from OpenAI. Status Code: {response.status_code}")

            response_data = response.json()
            ai_content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

            return ai_content
        
        except requests.RequestException as e:
            self.logger.error(f"Error connecting to OpenAI: {e}")
            raise OpenAIError(f"Error connecting to OpenAI: {e}")

        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise

# Create a function to pass the response to another AI step
def pass_to_next_step(response: str) -> str:
    # Placeholder for the next step logic
    # For this example, we'll simply log and return the response
    logger.debug(f"Passing response to next step: {response}")
    return response


class ThoughtProcess:
    """
    ThoughtProcess class handles the coordination and selection 
    of the appropriate questioning series based on conditions or inputs.
    """
    
    def __init__(self, chat_util):
        self.chat_util = chat_util
    
    def determine_series(self, user_input: str) -> Type:
        """
        Determine the appropriate questioning series based on user input or other conditions.
        
        Args:
        - user_input (str): Input from the user which helps in determining the series.
        
        Returns:
        - Type: The class type of the chosen questioning series.
        """
        # Simple keyword based determination, can be expanded with more advanced logic
        if "technical" in user_input:
            return TechnicalAssistance
        elif "feedback" in user_input:
            return FeedbackInquiry
        elif "product" in user_input:
            return ProductInquiry
        elif "entertainment" in user_input:
            return EntertainmentRecommendation
        elif "travel" in user_input:
            return TravelAssistance
        elif "about me" in user_input:
            return PersonalIntroduction
        else:
            return GeneralInquiry
    
    def execute_thought_process(self, user_input: str, username: str) -> Dict[str, str]:
        """
        Execute the chosen questioning series and return the assistant's responses.
        
        Args:
        - user_input (str): Input from the user.
        - username (str): Name of the user.
        
        Returns:
        - Dict[str, str]: Dictionary with questions as keys and AI responses as values.
        """
        series_class = self.determine_series(user_input)
        return series_class.execute_series(self.chat_util, username, user_input)  # added user_input


class OpenAIError(Exception):
    """Custom Exception class for OpenAI related errors."""
    pass

class GeneralInquiry:
    questions = [
        "Can you delve deeper into the topic of '{user_input}'?",
        "What are some broader contexts or implications related to '{user_input}'?",
        "How does '{user_input}' compare or relate to other similar topics or concepts?"
    ]

    @staticmethod
    def execute_series(chat_util, username: str, user_input: str) -> Dict[str, str]:
        return {q.format(user_input=user_input): chat_util.send_secondary_GPT4(q.format(user_input=user_input), username) for q in GeneralInquiry.questions}

class TechnicalAssistance:
    questions = [
        "What are the underlying principles or mechanisms of '{user_input}'?",
        "Are there known challenges or complications associated with '{user_input}'?",
        "How would you troubleshoot or approach issues related to '{user_input}'?"
    ]

    @staticmethod
    def execute_series(chat_util, username: str, user_input: str) -> Dict[str, str]:
        return {q.format(user_input=user_input): chat_util.send_secondary_GPT4(q.format(user_input=user_input), username) for q in TechnicalAssistance.questions}

class FeedbackInquiry:
    questions = [
        "What are some common feedback or opinions on '{user_input}'?",
        "How has '{user_input}' evolved or changed over time based on feedback?",
        "Are there alternative perspectives or viewpoints on '{user_input}'?"
    ]

    @staticmethod
    def execute_series(chat_util, username: str, user_input: str) -> Dict[str, str]:
        return {q.format(user_input=user_input): chat_util.send_secondary_GPT4(q.format(user_input=user_input), username) for q in FeedbackInquiry.questions}

class PersonalIntroduction:
    questions = [
        "How does '{user_input}' reflect personal experiences or values?",
        "What can '{user_input}' tell us about individual motivations or aspirations?",
        "How might different individuals interpret or perceive '{user_input}' differently?"
    ]

    @staticmethod
    def execute_series(chat_util, username: str, user_input: str) -> Dict[str, str]:
        return {q.format(user_input=user_input): chat_util.send_secondary_GPT4(q.format(user_input=user_input), username) for q in PersonalIntroduction.questions}

class ProductInquiry:
    questions = [
        "What are the key features or aspects of '{user_input}'?",
        "How does '{user_input}' compare to other products or solutions in the same category?",
        "What are the potential advantages or disadvantages of '{user_input}'?"
    ]

    @staticmethod
    def execute_series(chat_util, username: str, user_input: str) -> Dict[str, str]:
        return {q.format(user_input=user_input): chat_util.send_secondary_GPT4(q.format(user_input=user_input), username) for q in ProductInquiry.questions}

class EntertainmentRecommendation:
    questions = [
        "What are the thematic or stylistic elements of '{user_input}'?",
        "How might '{user_input}' resonate with different audiences?",
        "What are some lesser-known or niche aspects of '{user_input}'?"
    ]

    @staticmethod
    def execute_series(chat_util, username: str, user_input: str) -> Dict[str, str]:
        return {q.format(user_input=user_input): chat_util.send_secondary_GPT4(q.format(user_input=user_input), username) for q in EntertainmentRecommendation.questions}

class TravelAssistance:
    questions = [
        "What are the cultural, historical, or geographical significances of '{user_input}'?",
        "How might one prepare or plan for a trip related to '{user_input}'?",
        "What are some hidden gems or lesser-known facts about '{user_input}'?"
    ]

    @staticmethod
    def execute_series(chat_util, username: str, user_input: str) -> Dict[str, str]:
        return {q.format(user_input=user_input): chat_util.send_secondary_GPT4(q.format(user_input=user_input), username) for q in TravelAssistance.questions}

def main(user_input: str, username: str) -> dict:
    """Execute the thinking agent process based on passed user_input and username, then return responses."""
    
    # Initialize the ChatUtils instance
    chat_util_instance = ChatUtils()

    # Create a ThoughtProcess instance
    thought_process_instance = ThoughtProcess(chat_util_instance)

    # Execute the thought process based on user input
    responses = thought_process_instance.execute_thought_process(user_input, username)

    return responses

def display_responses(responses: dict) -> None:
    """Display the responses in a formatted manner."""
    for question, answer in responses.items():
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("-" * 50)

if __name__ == "__main__":
    # For simulation, get user input and username
    simulated_input = input("Enter your query (for simulation): ")
    simulated_username = input("Enter your username (for simulation): ")

    # Call the main function and display the results
    result_responses = main(simulated_input, simulated_username)
    display_responses(result_responses)
