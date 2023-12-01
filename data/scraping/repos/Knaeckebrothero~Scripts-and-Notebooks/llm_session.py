import os
import guidance
from dotenv import load_dotenv


# Class to manage a conversation
class Session:
    def __init__(self, llm_model: str = "gpt-3.5-turbo",
                 assistant_description: str = None, history: list[str] = None):
        # Load the API key
        load_dotenv()

        # Initialize the model
        self.llm_model = guidance.llms.OpenAI(
            llm_model, api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize the role
        if assistant_description is None:
            self.llm_role = "You are a helpful assistant."
        else:
            self.llm_role = assistant_description

        # Initialize the template
        self.template = "{{#system~}}{{system_prompt}}{{~/system}}"
        if history is not None:
            for i in range(0, len(history), 2):
                self.template += "{{#user~}}" + history[i] + "{{~/user}}"
                if i + 1 < len(history):
                    self.template += "{{#assistant~}}" + history[i + 1] + "{{~/assistant}}"

    # Method to generate a response
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        # Create the template
        generate_template = (
                self.template + """{{#user~}}{{prompt}}{{~/user}}{{#assistant~}}
                {{gen 'answer' temperature=""" + str(temperature) + " max_tokens="
                + str(max_tokens) + "}}{{~/assistant}}")

        print(generate_template)

        # Create the guidance program using the template and model
        generate_program = guidance.Program(generate_template, llm=self.llm_model)

        try:
            # Generate the answer
            generate_answer = generate_program(
                system_prompt=self.llm_role, prompt=prompt).variables().get('answer')

            # Add the messages to the template
            self.add_message_user(prompt)
            self.add_message_assistant(generate_answer)

            # Return the answer
            return generate_answer
        except Exception as e:
            # If there is an error, return the error
            generate_answer = f"ERROR: {e}"
        return generate_answer

    # Method to add a message from the user to the template
    def add_message_user(self, message: str):
        self.template += "{{#user~}}" + message + "{{~/user}}"

    # Method to add a message from the assistant to the template
    def add_message_assistant(self, message: str):
        self.template += "{{#assistant~}}" + message + "{{~/assistant}}"
