import os
import openai
from server.nlu.tf_model import YourTensorFlowModel

class NLPModel:
    def __init__(self):
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.tf_model = YourTensorFlowModel()

    def process_user_input(self, user_input, memory):
        """
        Process user input using either TensorFlow model or ChatGPT based on confidence.

        Parameters:
        user_input (str): The user input text.
        memory (dict): The memory data.

        Returns:
        tuple: response (str), memory_update (dict)
        """
        # Set up the OpenAI API client
        openai.api_key = self.api_key

        # Model Selector Logic
        use_openai = True
        if self.tf_model.get_confidence(user_input) > 0.8:
            use_openai = False

        # Processing Logic
        if use_openai:
            # Query ChatGPT for a response
            response_obj = openai.Completion.create(
                engine="text-davinci-003",
                prompt=user_input,
                max_tokens=150
            )
            response = response_obj['choices'][0]['text'].strip()
            memory_update = {}  # Assume no memory update or define logic to update memory
        else:
            # TensorFlow Model Processing
            response, memory_update = self.tf_model.generate_response(user_input, memory)

            # Active Selection
            prompt = user_input + " "
            responses = response.split("\n")
            active_response = app.config["active_selection"](self.tf_model, prompt, responses)
            response = active_response.strip()

        return response, memory_update

# Usage:
# nlp_model = NLPModel()
# response, memory_update = nlp_model.process_user_input("Hello!", {})