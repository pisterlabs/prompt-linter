import openai
import responses as r
import utilities as u
import mneumosyne as mn


class ThoughtProcess():
    def __init__(self):
        self.openAPIKey = u.import_text_file("../../openapikey.txt")
        # Replace 'YOUR_OPENAI_API_KEY' with your actual API key from OpenAI
        openai.api_key = self.openAPIKey

        # GPT instances representing different expertise
        # For processing text inputs
        self.language_instance = "text-davinci-002"
        # For processing audio inputs
        self.speech_instance = "text-davinci-003"

        # Context to hold shared information across GPT instances
        self.context = []

        self.short_term_memory = mn.ShortTermMemory()
        self.long_term_memory = mn.LongTermMemory()

    # Function to process text input using the appropriate GPT instance
    def process_input(self, input_text, context):
        if input_text.startswith("[AUDIO]"):
            instance = self.speech_instance
            # Remove the [AUDIO] prefix from audio input
            input_text = input_text[7:]
        else:
            instance = self.language_instance

        response = r.generate_gpt3_response(input_text,
                                            u.get_sentiment_score(input_text))

        openai.Completion.create(
            engine=instance,
            prompt=input_text,
            context=context,
            max_tokens=100  # Adjust max_tokens as needed
        )

        # Update the context with the response to maintain continuity
        context.append(response.choices[0].text.strip())  # type: ignore

        # Store the user input in short-term memory
        self.short_term_memory.encode_input(user_input, context)

        # Retrieve related memories from long-term memory
        # this object will need to be more complex
        context = "Current conversation"
        related_memories = self.long_term_memory.find_similar_memories(context)

        # Concatenate related memories and current input for full experience
        full_experience = user_input + " ".join(
            memory.content for memory in related_memories
            )

        # Process and respond to the full experience
        ai_response = self.process_input(full_experience, context)

        # Store the AI response in short-term memory
        self.short_term_memory.encode_input(ai_response, context)

        # Store the full experience in long-term memory
        self.long_term_memory.store_memory(context,
                                           full_experience,
                                           context)

        return response.choices[0].text.strip()  # type: ignore


thoughtLoop = ThoughtProcess()
# Main loop for the AI to continuously listen and respond
while True:
    user_input = input("User: ")
    response = thoughtLoop.process_input(user_input, thoughtLoop.context)
    print("AI:", response)
