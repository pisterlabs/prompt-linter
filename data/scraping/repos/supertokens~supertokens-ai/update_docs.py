import openai
import re
import os
import tiktoken
from dotenv import load_dotenv
import pandas as pd
from split_or_union_data_files import perform_split_or_union
load_dotenv()

root_dir = os.environ.get('DOCUMENTATION_PATH') or  ""
not_allowed = [root_dir + '/auth-react', root_dir + '/auth-react_versioned_docs', root_dir + '/auth-react_versioned_sidebars', root_dir + '/build', root_dir + '/change_me', root_dir + '/community', root_dir + '/node_modules', root_dir + '/nodejs', root_dir + '/nodejs_versioned_docs', root_dir + '/nodejs_versioned_sidebars', root_dir + '/website', root_dir + '/website_versioned_docs', root_dir + '/website_versioned_sidebars']
only_allow = [root_dir + '/mfa', root_dir + '/session', root_dir + '/emailpassword', root_dir + '/passwordless', root_dir + '/thirdparty']
consider_only_allow = False
# max_tokens_per_chunk = 500

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

chunks = [500, 1024, 2048]
df = {}
for max_tokens_per_chunk in chunks:
    embeddings_location = 'processed/' + str(max_tokens_per_chunk) + '-limit.csv'
    df[max_tokens_per_chunk] = pd.DataFrame(columns=['text', 'embeddings'])
    if os.path.exists(embeddings_location):
        df[max_tokens_per_chunk] = pd.read_csv(embeddings_location)

chunks_ignored = 0

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens):
    if (len(tokenizer.encode(text)) < max_tokens):
        return [text]
    global chunks_ignored

    # Split the text into sentences - some sentences end with a ". " and some with just a "." followed by a new line.
    sentences = re.split(r'\. |\.\n', text)

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(sentence + ". ")) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens and len(chunk) > 0:
            chunks.append(" ".join(chunk))
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            chunks_ignored += 1 # for analysis only
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence + ". ")
        tokens_so_far += token + 1

    return chunks

def find_df_for_text_from_existing_embeddings(text):
    for chunk in chunks:
        if chunk in existing_embeddings:
            for i in range(len(existing_embeddings[chunk])):
                if existing_embeddings[chunk].loc[i, 'text'] == text:
                    return existing_embeddings[chunk].loc[i]
    return None

def find_df_for_text_from_df(text):
    for chunk in chunks:
        for i in range(len(df[chunk])):
            if df[chunk].loc[i, 'text'] == text:
                return df[chunk].loc[i]
    return None


def convert_mdx_to_chunks(root_dir, chunk_size):
    mdx_content = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mdx'):
                filepath = os.path.join(subdir, file)
                
                continue_to_next_loop = False
                for i in not_allowed:
                    if i in filepath:
                        continue_to_next_loop = True
                if continue_to_next_loop:
                    continue

                continue_to_next_loop = True
                for i in only_allow:
                    if i in filepath:
                        continue_to_next_loop = False
                if continue_to_next_loop and consider_only_allow:
                    continue
                
                with open(filepath, 'r') as f:
                    contents = f.read()
                    if (len(tokenizer.encode(contents)) > chunk_size):
                        h2chunks = re.split(r'(?<=^##\s)', contents, flags=re.MULTILINE)
                        for chunk in h2chunks:
                            finalChunk_without_tags = chunk
                            finalChunk_without_tags = finalChunk_without_tags.replace("##", "")
                            finalChunk_without_tags = finalChunk_without_tags.replace(":::", "")
                            chunk_within_token_limit = split_into_many(finalChunk_without_tags, chunk_size)
                            for i in chunk_within_token_limit:
                                mdx_content.append(i)
                    else:
                        mdx_content.append(contents)
    return mdx_content
                    

# Set up OpenAI API credentials
openai.api_key = os.environ.get('OPEN_AI_KEY')

# Function to split the text into chunks of a maximum number of tokens
def to_token(text):
    return tokenizer.encode(text)

def get_embeddings(chunk_size):
    mdx_content = convert_mdx_to_chunks(root_dir, chunk_size)

    # convert the mdx_content list into a list tokens
    mdx_content_tokens = []
    for i in mdx_content:
        mdx_content_tokens.append(to_token(i))

    new_df = pd.DataFrame(columns=['text', 'embeddings'])
    curr_index = 0
    for i in range(len(mdx_content_tokens)):
        existing_df = find_df_for_text_from_existing_embeddings(mdx_content[i])
        if existing_df is not None:
            continue
        existing_df = find_df_for_text_from_df(mdx_content[i])
        if existing_df is not None:
            new_df.loc[curr_index, 'text'] = existing_df['text']
            new_df.loc[curr_index, 'embeddings'] = existing_df['embeddings']
            curr_index+=1
            continue
        print("=========================")
        print("Calculating embed for " + str(i) + " out of " + str(len(mdx_content_tokens)) + " for chunk size:" + str(chunk_size))
        print()
        embeddings = openai.Embedding.create(
            engine='text-embedding-ada-002',
            input=mdx_content_tokens[i]
        )['data'][0]['embedding']

        new_df.loc[curr_index, 'text'] = mdx_content[i]
        new_df.loc[curr_index, 'embeddings'] = embeddings
        curr_index+=1

    new_df.to_csv('processed/' + str(chunk_size) + '-limit.csv', index=False)
    return new_df

existing_embeddings = {}

for chunk in chunks:
    existing_embeddings[chunk] = get_embeddings(chunk)

perform_split_or_union()