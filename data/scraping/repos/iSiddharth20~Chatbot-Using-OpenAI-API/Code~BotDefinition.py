import openai

openai.api_key = "Enter Your API Key Here"

class OpenAIBot:
    def __init__(self,engine):
        # Initialize conversation with a system message
        self.conversation = [{"role": "system", "content": "You are a helpful assistant."}]
        self.engine = engine
    def add_message(self, role, content):
        # Adds a message to the conversation.
        self.conversation.append({"role": role, "content": content})
    def generate_response(self, prompt):
        # Add user prompt to conversation
        self.add_message("user", prompt)
        try:
            # Make a request to the API using the chat-based endpoint with conversation context
            response = openai.ChatCompletion.create( model=self.engine, messages=self.conversation)
            # Extract the response
            assistant_response = response['choices'][0]['message']['content'].strip()
            # Add assistant response to conversation
            self.add_message("assistant", assistant_response)
            # Return the response
            return assistant_response
        except:
            print('Error Generating Response!')