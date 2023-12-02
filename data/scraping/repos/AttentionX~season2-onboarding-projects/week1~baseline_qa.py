"""
A simple baseline for a question answering system.
"""
import os
from pathlib import Path
import yaml
import openai
from annoy import AnnoyIndex
from dotenv import load_dotenv
import guidance
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# --- load pre-processed chunks --- #
with open(Path(__file__).resolve().parent / "openai27052023.yaml", 'r') as f:
    paper = yaml.safe_load(f)
sentences = paper['sentences']


# --- embed chunks --- #
print("embedding chunks...")
embeddings = [
    r['embedding']
    for r in openai.Embedding.create(input=sentences, model='text-embedding-ada-002')['data']
] 

# --- index embeddings for efficient search (using Spotify's annoy)--- #
hidden_size = len(embeddings[0])
index = AnnoyIndex(hidden_size, 'angular')  #  "angular" =  cosine
for i, e in enumerate(embeddings): 
    index.add_item(i , e)
index.build(10)  # build 10 trees for efficient search

# --- iteratively answer questions (retrieve & generate) --- #
while True:
    query = input("Your question: ")
    embedding =  openai.Embedding.create(input = [query], model='text-embedding-ada-002')['data'][0]['embedding']
    # get nearest neighbors by vectors
    indices, distances = index.get_nns_by_vector(embedding,
                                                  n=3,  # return top 3
                                                  include_distances=True)

    guidance.llm = guidance.llms.OpenAI("text-davinci-003")
    # define the guidance program
    chitchat_detection = guidance(
    """
    title of the paper: {{title}}
    ---
    You are a Question Answering Machine. Your role is to answer the user's question relevant to the query.
    Your role is also to detect if the user's question is irrelevant to the paper. If the user says, for example,
    "hi, how are you doing?", "what is your name?", or "what is the weather today?", then the user's question is irrelevant.
    
    Now, judiciously determine if the following Query is relevant to the paper or not.

    Answer in the following format:
    Query: The question to be answered
    Reason: is the Query relevant to the paper? Show your reasoning. Explain why or why not.
    Final Answer: Either conclude with Yes (the Query is relevant) or No (the Query is irrelevant)
    ---
    Query: {{query}}
    Reasoning: {{gen "reasoning" stop="\\nF"}}
    Final Answer:{{#select "answer"}}Yes{{or}}No{{/select}}""")
    out = chitchat_detection(
        title=paper['title'],
        query=query,
    )
    answer = out['answer'].strip()
    # save your resources - don't answer if the question is irrelevant
    if answer == 'No':
        answer = "I'm afraid I can't answer your question because: "
        answer += f"{out['reasoning'].split('.')[0]}"
        print(answer)
        continue
    # if the question is relevant, proceed to answer
    results =  [ 
        (sentences[i], d)
        for i, d in zip(indices, distances)
    ]
    excerpts = [res[0] for res in results]
    excerpts = '\n'.join([f'[{i}]. \"{excerpt}\"' for i, excerpt in enumerate(excerpts, start=1)])
    # first, check if the query is answerable 
    # proceed to answer
    prompt = f"""
    user query:
    {query}
    
    title of the paper:
    {paper['title']}
    
    excerpts: 
    {excerpts}
    ---
    given the excerpts from the paper above, answer the user query.
    In your answer, make sure to cite the excerpts by its number wherever appropriate.
    Note, however, that the excerpts may not be relevant to the user query.
    """
    # uses gpt-3.5-turbo 
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                messages=[{"role": "user", "content": prompt}])
    answer = chat_completion.choices[0].message.content
    answer += f"\n--- EXCERPTS ---\n{excerpts}"
    print(answer)




