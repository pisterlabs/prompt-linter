import openai
import tiktoken # Used to count tokens

# Load OpenAI API key from file
with open("openai_api_key.txt", "r") as f:
    openai.api_key = f.read()

# Limit on number of GPT3 tokens per request
TOKEN_LIMIT = 2000

BASE_PROMPT = '''
                Below I will give you a video transcription.
                Please convert this transcription into a "blog post" summary, highlighting the key snippets and content.
                Make sure to use Markdown Format to write the post (.md), including headers, a table of contents, and good punctuation.\n
                '''

def generateBlog(transcript):
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    prmt = BASE_PROMPT + transcript
    tokens = tokenizer.encode(prmt)
    token_count = len(tokens)
    print("-- Token count: {} / {}".format(token_count, TOKEN_LIMIT))
    if token_count > TOKEN_LIMIT:
        return "Error: Transcript too long. Please use a shorter video.", 400
    # print(prmt)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prmt,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"],
    )
    print("Response: {}".format(response))
    return response.choices[0].text
