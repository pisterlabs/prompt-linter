import openai
import os

# Set up OpenAI API credentials
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define function to process user input and generate bot response
def process_input(user_input, chat_history=None):
    # Combine previous conversation history with new user input
    prompt = ""
    if chat_history is not None:
        prompt += chat_history + "\n"
    prompt += user_input + "\n"

    # Send combined prompt to OpenAI API for processing
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract bot response from OpenAI API output
    bot_response = response.choices[0].text.strip()

    return bot_response

# Define main function to run chatbot
def run_chatbot():
    print("Welcome to the chatbot!")
    
    # Initialize empty conversation history
    chat_history = None
    
    while True:
        # Get user input from command line
        user_input = input("> ")

        # Process user input and generate bot response using conversation history as context
        bot_response = process_input(user_input, chat_history)

        # Display bot response in command line
        print("Bot: " + bot_response)

        # Ask user for feedback on bot response and update model based on feedback using reinforcement learning
        feedback = input("Was that helpful? (y/n): ")
        
        if feedback.lower() == "y":
            openai.Classification.create(
                model="text-davinci-002",
                examples=[["User: " + user_input, "Bot: " + bot_response]],
                labels=["good"],
            )
        else:
            openai.Classification.create(
                model="text-davinci-002",
                examples=[["User: " + user_input, "Bot: " + bot_response]],
                labels=["bad"],
            )

        # Update conversation history with new input and output 
        if chat_history is not None:
            chat_history += "\n" + user_input + "\n" + bot_response
        else:
            chat_history = user_input + "\n" + bot_response

# Call main function to start chatbot
if __name__ == "__main__":
    run_chatbot()
