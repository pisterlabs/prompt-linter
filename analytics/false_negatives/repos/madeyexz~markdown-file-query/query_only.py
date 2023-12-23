import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import os, sys, json

# This file is a trimmed and slightly-altered version of main.py

def pinecone_init(index_name: str = 'notion-database'):
    '''initialize connection to pinecone (get API key at app.pinecone.io)'''
    # index_name = 'notion-database' # assigned as the default value
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment="us-east1-gcp"
    )

    # check if index already exists (it shouldn't if this is first time)
    if index_name in pinecone.list_indexes():
        index = pinecone.Index(index_name)
        return index
    else:
        print(f"index {index_name} not found")
        quit()

def get_docs(path: str = 'docs.json'):
    '''get indexed docs from docs.json, which is a memory file of main.py'''
    with open(path, 'r') as f:
        docs = json.load(f)
    return docs


def pinecone_query(query: str = "who are you", docs=get_docs(), index=pinecone_init()):
    query_coord = OpenAIEmbeddings().embed_query(query)
    # retrieve from Pinecone
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
    # print(prompt.format(question=query, contents=contents_str)) # for debugging purpose
    answer = chain.run(
        question=query,
        contents=contents_str,
        #verbose=True
        )
    return answer

def ans_cont_to_file(answer, contents_str):
    # This last set of code is to write the answer and contents to text files.
    with open ("answer.txt", "w") as f:
        f.write(answer)
    with open ("contents.txt", "w") as h:
        h.write(contents_str)

def main():
    try:
        query = sys.argv[1]
    except IndexError:
        query = input("ask a question: ")
    
    print("connecting to pinecone index...")
    index = pinecone_init("notion-database")
    print("getting docs")
    docs = get_docs()
    
    print("querying pinecone...")
    # query = input("ask a question")
    
    contents_str =  pinecone_query(query, docs, index)
    print("querying gpt...")
    answer = ask_gpt3(query=query, contents_str=contents_str)
    
    # optimal, converts the answer and contents to text files
    ans_cont_to_file(answer, contents_str)
    
    print(f"done! the answer to '{query}' is: '{answer}'")


if __name__ == "__main__":
    main()