import openai
import tiktoken

# Set up the OpenAI API key
openai.api_key = "YOUR_API_KEY"

# Define the prompt message
prompt = "Hi GPT, I will give you a textual input consisting of a markdown file. Please note that I will supply the next chunk after the non-last files. This was the last chunk. Thank you!"

# Read the markdown file into a string variable
with open('path/to/markdown/file.md', 'r') as f:
    text = f.read()

# Calculate the number of tokens in the text
token_count = tiktoken.count_tokens(text)

# Define the chunk size and token limit
chunk_size = 5  # number of paragraphs per chunk
token_limit = 4096  # maximum number of tokens per chunk

# Split the text into paragraphs
paragraphs = text.split('\n\n')

# Chunk the paragraphs
chunks = []
chunk = ''
token_count = 0
for paragraph in paragraphs:
    # Add the paragraph to the current chunk
    chunk += paragraph + '\n\n'
    # Update the token count
    token_count += tiktoken.count_tokens(paragraph)
    # If the token count exceeds the limit, start a new chunk
    if token_count > token_limit:
        # Generate the next chunk using OpenAI
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=token_limit,
            temperature=0.5,
            stop=None,
            n=1,
            timeout=60,
        )
        next_chunk = response.choices[0].text
        chunks.append(chunk + next_chunk)
        chunk = ''
        token_count = 0
    # If the chunk size exceeds the limit, start a new chunk
    if len(chunks) == chunk_size:
        # Generate the next chunk using OpenAI
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=token_limit,
            temperature=0.5,
            stop=None,
            n=1,
            timeout=60,
        )
        next_chunk = response.choices[0].text
        chunks.append(chunk + next_chunk)
        chunk = ''
        token_count = 0

# Add the last chunk
if chunk:
    chunks.append(chunk)

# Add the last message to the last chunk
chunks[-1] += "This was the last chunk."

# Write the chunks to separate files
for i, chunk in enumerate(chunks):
    with open(f'path/to/output/chunk_{i}.md', 'w') as f:
        # Add the message to the non-last chunks
        if i < len(chunks) - 1:
            chunk += "Please note that I will supply the next chunk after the non-last files."
        f.write(chunk)