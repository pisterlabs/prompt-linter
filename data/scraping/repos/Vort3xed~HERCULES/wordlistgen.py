import openai

def generate_word_list(network_name, output_file):
    # Set up your OpenAI API credentials
    openai.api_key = 'ENTER API KEY HERE'

    # Specify the prompt
    prompt = f'Create a word list where each word has its own line for the network name: "{network_name}".\n\nWord List:'

    # Generate the word list using OpenAI's Chat Completions API
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        n=10,
        stop=None,
        temperature=0.7
    )

    # Extract the generated word list from the API response
    choices = response.choices
    if len(choices) > 0:
        word_list = choices[0].text.strip().split('\n')
    else:
        word_list = []

    # Write the word list to a text file
    with open(output_file, 'w') as file:
        file.write('\n'.join(word_list))
        file.flush()

    print(f"Word list generated and saved to {output_file}")