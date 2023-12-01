import os # to get system environment variables
import sys # to parse system arguments
import json # to dump and load data (lists) elegently
import pinecone # vector database service, core of the VDB query
from pathlib import Path # used to manipulate file paths elegently
from tqdm.auto import tqdm # to show progress bar
import time # to avoid RateLimitError
from langchain.text_splitter import CharacterTextSplitter # to split texts
from langchain.prompts import PromptTemplate # makes query easier 
from langchain.llms import OpenAI # to query LLMs
from langchain.chains import LLMChain # makes query easier
from langchain.embeddings import OpenAIEmbeddings # to turn texts into vectors

def pinecone_init(index_name: str = 'notion-database'):
    '''initialize connection to pinecone (get API key at app.pinecone.io)'''
    # index_name = 'notion-database' # assigned as the default value
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment="us-east1-gcp"
    )

    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine',
            # metric='euclidean',
            metadata_config={'indexed': ['channel_id', 'published']} # useless code, guess why im not deleting this yet?
        )
    # connect to index
    index = pinecone.Index(index_name)
    # view index status with
    # index.describe_index_stats()
    return index

    # an error was met and solved upon retring && upgrading jupyter notebook with `pip install notebook --upgrade`

def md_digest(ps: list = list(Path("Notion_DB/").glob("**/*.md"))):
    '''This is the logic for ingesting Notion data into LangChain.'''
    
    # Here we load in the data in the format that Notion exports it in.
    data = []
    sources = []
    for p in ps:
        with open(p) as f:
            data.append(f.read())
        sources.append(p)

    # We do this due to the context limits of the LLMs.
    # chunk size is 1000, which means each chunk of text will be 1000 characters long, and that the separator is a new line
    text_splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")
    docs = []
    metadatas = []
    for i, d in enumerate(data):
        # where i, d is the index and content of each .md file respectively
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": sources[i]}] * len(splits))

    # after digestion, we save the docs to local json files for later queries to avoid re-encoding.
    with open('docs.json', 'w') as f:
        json.dump(docs, f)    
    
    return docs
    # question, will the data be too big/unspecific for each chunk?
    # now len(docs) should be the number of vectors this is going to create

def pinecone_upload(docs: list = md_digest(), index=pinecone_init()):
    '''This is the logic for uploading the data into Pinecone.'''
    # upload to pinecone

    id_batch = [str(x) for x in range(0, len(docs))]
    coord_list = []

    for i in tqdm(range(0, len(docs))):
        # this line is added to avoid RateLimitError, where 60 second is a very random but conservative number.
        # a stupid approach by me :D
        rest = 60
        if i != 0 and i % 60 == 0:
            print(f"let's wait for {rest} seconds to avoid RateLimitError... \(since im not a paid user\))")
            for i in tqdm(range(0, rest)):
                time.sleep(1)

        # get texts to encode
        texts = docs[i]
        coord = OpenAIEmbeddings().embed_query(texts)
        coord_list.append(coord)

    # prepare and upload the vectors to Pinecone
    vectors = list(zip(id_batch, coord_list))
    index.upsert(vectors)


def pinecone_query(query: str = "who are you", docs=md_digest(), index=pinecone_init()):
    query_coord = OpenAIEmbeddings().embed_query(query)
    # retrieve from Pinecone
    # get relevant contexts (including the questions)
    query_res = index.query(query_coord, top_k=3, include_metadata=True)

    content_ids = [
            int(x['id']) for x in query_res['matches']
        ]
    contents = [docs[i] for i in content_ids]
    contents_str = "\n\n".join(contents)
    
    return contents_str


def ask_gpt3(query:str ="who are you",contents_str=pinecone_query()):
    
    prompt = PromptTemplate(
        input_variables=["question","contents"],
        template=''' Answer this question: "{question}" using the contents below
        Contents:
        {contents}
        Answer:
        ''',
    )

    chain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=prompt,
        # verbose=True,
        )

    answer = chain.run(question=query,contents=contents_str)
    return answer

def ans_cont_to_file(answer, contents_str):
    # This last set of code is to write the answer and contents to text files.
    with open ("answer.txt", "w") as f:
        f.write(answer)
    with open ("contents.txt", "w") as h:
        h.write(contents_str)

def main():
    print("initiating pinecone index...")
    index = pinecone_init("notion-database")
    directory, query = sys.argv[1], sys.argv[2]
    
    print("digesting docs...")
    docs = md_digest(list(Path(directory).glob("**/*.md")))
    # docs = md_digest()
    
    print("uploading datas to pinecone...")
    pinecone_upload(docs, index)
    
    print("querying pinecone...")
    # query = input("ask a question")
    contents_str =  pinecone_query(query, docs)
    
    print("querying gpt...")
    answer = ask_gpt3(query=query, contents_str=contents_str)
    
    # optimal, converts the answer and contents to text files
    print("writing results to answer.txt and contents.txt")
    ans_cont_to_file(answer, contents_str)
    
    print(f"done! the answer to '{query}' is: '{answer}'")

if __name__ == "__main__":
    main()