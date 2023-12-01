import openai
import time

# Set up the OpenAI API client
openai.api_key = "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"
max_time = 1

print("Hello! I am Assistant, a large language model trained by OpenAI. What can I help you with?")

while True:
    # Get input from the user
    user_input = input("Enter your message: ")

    # Use the OpenAI API to generate a response, with a timeout of 10 seconds
    start_time = time.time()
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=user_input,
        max_tokens=1024,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    end_time = time.time()

    # Check if the response took longer than 10 seconds to generate
    elapsed_time = end_time - start_time
    if elapsed_time > max_time:
        print("Sorry, I was unable to generate a response in time. Please try again.")
    else:
        # Print the response
        print(response["choices"][0]["text"])
