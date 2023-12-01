import openai
import requests

# Set your OpenAI API key
openai.api_key = 'sk-nnn'


# Define a prompt related to software development
prompt = "Write a blog post about the importance of version control in software development."

# Generate content using ChatGPT
response = openai.Completion.create(
    engine="text-davinci-003",  # Use the appropriate GPT-3 model name
    prompt=prompt,
    max_tokens=150
)

# Get the generated content from the response
generated_content = response.choices[0].text.strip()

# Print the generated content
print("Generated Content:\n", generated_content)

# Generate tags using ChatGPT
tags_prompt = "Generate 5 tags related to software development:"
tags_response = openai.Completion.create(
    engine="davinci-codex",
    prompt=tags_prompt,
    max_tokens=30 * 5  # Adjust the max_tokens for 5 tags
)

# Extract the generated tags from the response
generated_tags = tags_response.choices[0].text.strip().split("\n")

# Print the generated tags
print("Generated Tags:\n", generated_tags)

# Medium API endpoint for creating a post
MEDIUM_API_URL = f'https://api.medium.com/v1/users/777/posts'

# Create and post a new article on Medium
def create_post_on_medium(title, content, tags):
    headers = {
        'Authorization': f'Bearer {YOUR_MEDIUM_API_TOKEN}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    payload = {
        'title': title,
        'contentFormat': 'html',
        'content': content,
        'tags': tags,
        'publishStatus': 'public'
    }

    response = requests.post(MEDIUM_API_URL, headers=headers, json=payload)

    if response.status_code == 201:
        print("Successfully created and published a post on Medium!")
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    title = "The Importance of Version Control in Software Development"
    create_post_on_medium(title, generated_content, generated_tags)
