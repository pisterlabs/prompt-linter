# openai.api_key = "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"
import asyncio
import openai

# Set up the OpenAI API client
openai.api_key = "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"

async def generate_response(prompt):
    # Use the OpenAI API to generate a response, with a timeout of 10 seconds
    try:
        response = await asyncio.wait_for(
            openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=1024,
                temperature=0.5,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            ),
            timeout=10
        )
        return response["choices"][0]["text"]
    except asyncio.TimeoutError:
        return "Sorry, I was unable to generate a response in time. Please try again."

async def main():
    print("Hello! I am Assistant, a large language model trained by OpenAI. What can I help you with?")

    while True:
        # Get input from the user
        user_input = input("Enter your message: ")

        # Generate a response asynchronously
        response = await generate_response(user_input)

        # Print the response
        print(response)

# Run the main function
asyncio.run(main())
