import openai
from pymilvus import MilvusClient
import re, os, hashlib, fnmatch
from tqdm.auto import tqdm
from time import sleep
import datetime

# global variables
OPENAI_EMBED_MODEL = "text-embedding-ada-002"
openai.api_key = os.getenv("OPENAI_API_KEY")
MILVUS_API_KEY = os.getenv("MILVUS_API_KEY")
MILVUS_URI = "https://in03-1e1cf095ffcfaac.api.gcp-us-west1.zillizcloud.com"
MILVUS_COLLECTION_NAME = "obsidian_second_brain"

def search_image_files(filename, directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            if f == filename:
                ext = os.path.splitext(f)[1].lower()
                if ext in ('.png', '.jpg', '.jpeg', '.gif'):
                    return os.path.join(dirpath, f)
    return None

def get_all_attachments_in_text(note_string):
    regex_pattern = r"\[\[.*?\]\]|!\[\[.*?\]\]"
    strip_char = r"\[|\]|!"
    matches = [re.sub(strip_char,'',match) for match in re.findall(regex_pattern,  note_string) ]
    return matches

def remove_urls(text):
    # Regular expression pattern for matching URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Remove URLs from the text
    without_urls = re.sub(url_pattern, '', text)
    return without_urls

def remove_obsidian_links(text):
    clean = re.compile('\[\[.*?\]\]|!\[\[.*?\]\]')
    return re.sub(clean, '', text)

def parse_markdown_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    _, filename = os.path.split(file_path)

    # Remove the YAML frontmatter.
    # Initialize a counter to keep track of how many '---'s we've seen
    count = 0
    for i, line in enumerate(lines):
        # If we've seen one '---' already and we've just seen another, return the rest of the lines
        if line == '---\n' and count == 1:
            lines = lines[i:]
            break
        # If we've just seen our first 'b', increment the count
        elif line == '---\n' and count == 0:
            count += 1

    # insert the filename as first element
    lines.insert(0, filename + '\n')
    return lines

def split_list(note_lines_list, word_threshold):
    sublists = []
    sublist = []
    subtotal = 0
    for sentence in note_lines_list:
        sublist.append(sentence)
        word_count = len(sentence.split())
        if subtotal + word_count > word_threshold:
            sublists.append(sublist)
            sublist = []
            subtotal = 0
        subtotal += word_count
    if sublist:
        sublists.append(sublist)
    return sublists


def get_files_to_index(rootdir):
    searchstr = 'publish: true'
    files_to_index = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if fnmatch.fnmatch(file, '*.md') == True:
                filepath = os.path.join(subdir, file)
                if fnmatch.fnmatch(filepath,'*/.trash/*') == False:
                    with open(filepath, 'r') as f:
                        if searchstr in f.read():
                            files_to_index.append(filepath)
    return files_to_index

def initialize_milvus_client():
    # Initialize a MilvusClient instance
    # Replace uri and token with your own
    client = MilvusClient(
        uri=MILVUS_URI,
        # - For a serverless cluster, use an API key as the token.
        # - For a dedicated cluster, use the cluster credentials as the token
        # in the format of 'user:password'.
        token=MILVUS_API_KEY
    )
    return client 

def create_note_snippets(files_list):
    VECTOR_WORD_LIMIT = 800
    notes_snippets = []
    for file in files_list:
        note_lines = parse_markdown_file(file)        
        note_text_split = split_list(note_lines, VECTOR_WORD_LIMIT)
        for i,note_text_split_snippet in enumerate(note_text_split):
            note_snippet = "".join(note_text_split_snippet)
            notes_snippets.append({
                'uuid': hashlib.sha256((file + "_^_" + str(i)).encode()).hexdigest(),
                'file': file,
                'section': i,
                'note': note_snippet
            })

    print(f"Adding {len(notes_snippets)} chunks from {len(files_list)} files")
    return notes_snippets

def upload_to_milvus(notes_snippets, milvus_client):
    batch_size = 100  # how many embeddings we create and insert at once

    for i in tqdm(range(0, len(notes_snippets), batch_size)):
        # find end of batch
        i_end = min(len(notes_snippets), i+batch_size)
        meta_batch = notes_snippets[i:i_end]
        # get ids
        ids_batch = [x['uuid'] for x in meta_batch]
        # get notes to encode
        notes = [x['note'] for x in meta_batch]
        # get file names to encode
        files = [x['file'] for x in meta_batch]
        # create embeddings (try-except added to avoid RateLimitError)
        try:
            res = openai.Embedding.create(input=notes, engine=OPENAI_EMBED_MODEL)
        except Exception as e:
            # handle the exception by printing a message
            print(f"An exception occurred: {repr(e)}")
            done = False
            while not done:
                #sleep(5)
                try:
                    res = openai.Embedding.create(input=notes, engine=OPENAI_EMBED_MODEL)
                    done = True
                except Exception as e:
                    print(f"Still getting an exception: {e} ... Passing")
                    print(files)
                    pass
        embeds = [record['embedding'] for record in res['data']]

        # cleanup metadata
        meta_batch = [{
            'uuid': x[0]['uuid'],
            'file': x[0]['file'],
            'note': x[0]['note'],
            'vector': x[1]
        } for x in zip(meta_batch, embeds)]

        # add data to milvus
        # Insert multiple entities
        res = milvus_client.insert(
        collection_name=MILVUS_COLLECTION_NAME,
        data=meta_batch
        )
    print(f"Finished upserting to Milvus index: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    milvus_client = initialize_milvus_client()
    files_list = get_files_to_index(rootdir="/Users/ammarh/Documents/second-brain/")
    notes_snippets = create_note_snippets(files_list=files_list)
    upload_to_milvus(notes_snippets, milvus_client)
