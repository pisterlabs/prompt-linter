# imports
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search









# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

openai.api_key = "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"

embeddings_path = "Book1.csv"



# 'list of people working in Acumen:  David Avetisyan Co-Founder and the director: , Armen Khachatryan - software developer likes playing fooball,  Marlena Mirzoyan - Project Manager enjoys playing a game chto gde kogda, Anna Snkhchyan - .net developer does great with one of the big customers of Acumen: Cinchy,  Sophie Mehrabyan Co-Founder she lives in Barcelona, Suren Mardanyan developer he likes playing football, Hripsime Manukyan react developer',

MAX_TOKENS = 1600
w_strings = [
                      """list of people working in Acumen: 

David Avetisyan
Co-Founder
Connect on LinkedIn : https://www.linkedin.com/in/davidavetisyan-acutech/
David is an IT expert with 15+ years of experience. He has led cross-functional engineering teams, delivering high-quality solutions to customers in North America, Europe and Armenia. As a Co-Founder/Director at Acumen, David oversees the company’s strategy and development of Acumen’s professional software services portfolio, ensuring the quality and efficiency of all work completed by Acumen Delivery and Engineering staff.


Sophie Mehrabyan
Co-founder
Connect on LinkedIn : https://www.linkedin.com/in/sophiemehrabyan/

Sophie is an innovative professional with 20+ years of management experience in IT business with demonstrated initiative, creativity and success. She has an excellent leadership, management, communication, interpersonal, intuitive, and analytical skills. Sophie sets the future direction of Acumen. As a board member of UATE Sophie is working with the IT community to advance the technology sector in Armenia.


Marlena Mirzoyan
Project Manager

Ani Baghyan
Senior Software Engineer

Sosy Vardanyan
HR Generalist

Seda Gevorgyan
Project Manager

Mher Vahramyan
Software Engineer

Denis Nekrasov
Software Engineer

Hripsime Manukyan
Senior Software Engineer

Vardan Manukyan
Senior QA Engineer

Toma Arakelyan
QA Engineer

Ruben Ghandilyan
Software Engineer

Simon Simonyan
Senior QA Engineer

Ashot Minasyan
Senior Software Engineer

Konstantin Shaimardanov
Software Engineer

Armen Khachatryan
Senior Software Engineer


All the employees are based in Yerevan except Marlena who is currently working from the USA
""",

"""
Salaries:

Marlena Mirzoyan
400000
 

Ani Baghyan
300000

Sosy Vardanyan
300000

Seda Gevorgyan
100000

Mher Vahramyan
100000

Denis Nekrasov
100000

Hripsime Manukyan
100000

Vardan Manukyan
100000

Toma Arakelyan
200000

Ruben Ghandilyan
300000

Simon Simonyan
400000

Ashot Minasyan
600000

Konstantin Shaimardanov
300000

Armen Khachatryan
100000
"""
]
 





BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

embeddings = []
for batch_start in range(0, len(w_strings), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = w_strings[batch_start:batch_end]
    print(f"Batch {batch_start} to {batch_end-1}")
    response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response["data"]):
        assert i == be["index"]  # double check embeddings are in same order as input
    batch_embeddings = [e["embedding"] for e in response["data"]]
    embeddings.extend(batch_embeddings)

df = pd.DataFrame({"text": w_strings, "embedding": embeddings})


#df = pd.read_csv(embeddings_path)
# convert embeddings from CSV str type back to list type
#df['embedding'] = df['embedding'].apply(ast.literal_eval)


# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]




def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below information about Acumen Technologies software company to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nInformation:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Acumen."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

print('Ask anything about Acumen')
input1 = input()
while(input1!='exit'):
    x = ask(input1)
    print(x)
    print("\n")
    input1 = input()
 