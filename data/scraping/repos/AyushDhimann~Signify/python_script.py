import sys
import openai

# Replace with your OpenAI API key
api_key = "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"
openai.api_key = api_key

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python_script.py <user_message>')
        sys.exit(1)

    # Read the contents of the text file
    with open('output.txt', 'r') as file:
        file_contents = file.read()

    # Get the user message from the command-line argument
    user_message = sys.argv[1]

    # Combine the file contents and user message as the prompt for ChatGPT
    prompt = f"{file_contents}\nUser: {user_message}"

    # Implement your logic here
    # Use the combined prompt for ChatGPT
    response = openai.Completion.create(
        engine="text-davinci-002",  # Replace with your preferred ChatGPT engine
        prompt=prompt,
        max_tokens=100,  # Adjust token limit as needed
    )

    bot_response = response.choices[0].text.strip()

    # Print the bot response to stdout for Express.js to read
    print(bot_response)
