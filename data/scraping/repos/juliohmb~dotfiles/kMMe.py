import openai

# Set up the OpenAI API client
openai.api_key = "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"

print("Hello! I am Assistant, a large language model trained by OpenAI. What can I help you with?")

while True:
    # Get input from the user
    user_input = input("Enter your message: ")

    # Use the OpenAI API to generate a response
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=user_input,
        max_tokens=1024,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Print the response
    print(response["choices"][0]["text"])
