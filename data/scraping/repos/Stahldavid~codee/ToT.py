import logging
from queue import Queue
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define constants for evaluation criteria
MAX_EVALUATIONS = 11  # Maximum number of evaluations
EVALUATION_SCORE_THRESHOLD = 7  # Threshold for evaluation score

# Import modules for creating message templates and data structures
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# Create message templates for each function using SystemMessagePromptTemplate and HumanMessagePromptTemplate classes
decomposition_system_template = SystemMessagePromptTemplate.from_template(
    "You are an AI that decomposes code generation tasks."
)
decomposition_human_template = HumanMessagePromptTemplate.from_template(
    "Decompose the code generation task: {task}"
)

generation_system_template = SystemMessagePromptTemplate.from_template(
    "You are an AI that generates possible next steps for code."
)
generation_human_template = HumanMessagePromptTemplate.from_template(
    "Generate a possible next step for the code: {code}"
)

evaluation_system_template = SystemMessagePromptTemplate.from_template(
    "You are an AI that evaluates the quality of proposed code on a scale from 0 to 10. Just responde score : x"
)
evaluation_human_template = HumanMessagePromptTemplate.from_template(
    "Evaluate the quality of the proposed code on a scale from 0 to 10.: {code}"
)

search_system_template = SystemMessagePromptTemplate.from_template(
    "You are an AI that chooses the next step to take from proposed next steps."
)
search_human_template = HumanMessagePromptTemplate.from_template(
    "From the proposed next steps, choose the next step to take: {proposals}"
)

# Import modules for parsing output and validating data models
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

# Define a data model for evaluation score using Pydantic and a custom validator
class EvaluationScore(BaseModel):
    score: int = Field(description="the evaluation score as an integer between 0 and 10")

    @validator('score')
    def score_must_be_in_range(cls, field):
        if field < 0 or field > 10:
            raise ValueError("Score must be between 0 and 10!")
        return field

# Create an output parser for the evaluation chain using PydanticOutputParser and the EvaluationScore data model
evaluation_parser = PydanticOutputParser(pydantic_object=EvaluationScore)

class EvaluationChain(LLMChain):
    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        
        # Parse the result as a dictionary and get the value of the score key
        result_dict = evaluation_parser.parse(result)
        score_value = result_dict["score"]
        
        # Convert the score value to an integer and return it
        int_result = int(score_value)
        return int_result


# Set up ChatOpenAI models for each function with different temperature and model name parameters
decomposition_llm = ChatOpenAI(temperature=0.3, model_name='gpt-3.5-turbo')
generation_llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
evaluation_llm = ChatOpenAI(temperature=0.5, model_name='gpt-3.5-turbo')
search_llm = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo')

# Set up LLMChains for each function using the ChatOpenAI models and the message templates
decomposition_chain = LLMChain(llm=decomposition_llm, prompt=ChatPromptTemplate.from_messages([decomposition_system_template, decomposition_human_template]))
generation_chain = LLMChain(llm=generation_llm, prompt=ChatPromptTemplate.from_messages([generation_system_template, generation_human_template]))
evaluation_chain = EvaluationChain(llm=evaluation_llm, prompt=ChatPromptTemplate.from_messages([evaluation_system_template, evaluation_human_template]))
search_chain = LLMChain(llm=search_llm, prompt=ChatPromptTemplate.from_messages([search_system_template, search_human_template]))

# Define a helper function to validate the task input as a non-empty string
def validate_task(task):
    if not task or not isinstance(task, str) or len(task.strip()) == 0:
        logging.error("Invalid task. Please provide a non-empty string.")
        return False
    return True

# Define a helper function to process the task input using the LLMChains and update the code output and evaluation counter
def process_task(task, queue, stack, code, evaluation_counter):
    try:
        # Decompose the task into smaller parts using the decomposition chain
        decomposition = decomposition_chain(task)
        
        # For each part, generate a possible next step using the generation chain
        for part in decomposition:
            generation = generation_chain(part) 
            
            # Evaluate the quality of the generated code using the evaluation chain and the output parser
            evaluation_score = evaluation_chain(generation) 
            evaluation_counter += 1
            
            # If the evaluation score is above the threshold, add the generated code to the queue and stack
            if evaluation_score >= EVALUATION_SCORE_THRESHOLD:
                queue.put(generation)
                stack.append(generation)
            # If the evaluation score is below the threshold, revert to the last state in the stack
            elif stack:
                last_state = stack.pop()
                queue.put(last_state)
        
        # If the queue is not empty, choose the next step to take from the proposed next steps using the search chain
        if not queue.empty():
            search = search_chain(queue.queue)
            code += search
    except Exception as e:
        logging.error(f"An error occurred: {e}")

    # Clear the queue and stack for the next iteration
    queue.queue.clear()
    stack.clear()

    return code, evaluation_counter

# Define a helper function to check if the code is complete by looking for a special marker
def code_complete(code):
    return code.endswith("end of code")

# Define the main function that takes a task input and produces a code output using the LLMChains
def main():
    # Get the task input from the user or a predefined variable
    task = 'your task here'
    
    # Validate the task input
    if not validate_task(task):
        return

    # Initialize a queue and a stack to store intermediate results
    queue = Queue()
    stack = []
    
    # Add the task input to the queue and stack
    queue.put(task)
    stack.append(task)

    # Initialize an empty string for the code output
    code = ""
    
    # Initialize a counter for the number of evaluations done
    evaluation_counter = 0

    # Loop until the queue is empty or the maximum number of evaluations is reached
    while not queue.empty():
        # Get the next task from the queue
        task = queue.get()
        
        # Process the task using the LLMChains and update the code output and evaluation counter
        code, evaluation_counter = process_task(task, queue, stack, code, evaluation_counter)

        # Check if the maximum number of evaluations is reached
        if  evaluation_counter >= MAX_EVALUATIONS:
            break
    
    # Log the final code output
    logging.info(f"Final code: {code}")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
