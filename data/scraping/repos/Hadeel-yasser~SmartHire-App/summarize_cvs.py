
import os
import openai

# Set up your OpenAI configuration here
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_base = "https://trippee.openai.azure.com"  # Your Azure OpenAI resource's endpoint value.
openai.api_key = "1b9551b253a44d549f9f610a17dc2e22"

def summarize_text(candidate_id,txt_file_path_dict):
    # Read text from the input file
    text_path = txt_file_path_dict[candidate_id]
    with open(text_path, "r") as text_file:
        text_content = text_file.read()

    # Create a message for OpenAI
    messages = [
        {
            "role": "system",
            "content": "Assistant is an intelligent chatbot designed to summarize the contents of the text provided." 
            "Don't mention in the answer anything about from where the text was extracted or provided. Just directly start summarizing the cv"
        },
        {
            "role": "user",
            "content": text_content
        }
    ]

    # Send the message to OpenAI for summarization
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",  # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
        messages=messages
    )

    return response['choices'][0]['message']['content']

# Example usage:
'''text_file_path = "E:\\ITworx\\CVs\\Documents\\software_engineer\\Hadeel's_CV.txt"  # Replace with the actual text file path

answer = summarize_text(text_file_path)

print("\nAnswer:")
print(answer)'''
