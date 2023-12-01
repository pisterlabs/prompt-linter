import functions_framework
import os
from google.cloud.sql.connector import Connector
import sqlalchemy
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from google.cloud import storage
import time

try:
    model = SentenceTransformer("/tmp/all-MiniLM-L6-v2")
except:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.save("/tmp/all-MiniLM-L6-v2")

DB_INSTANCE = os.environ.get("INSTANCE")
DB_USER = os.environ.get("USER")
DB_PASS = os.environ.get("PASS")
DB_NAME = os.environ.get("NAME")
CLAUDE_KEY = os.environ.get("CLAUDE_KEY")
K_NEAREST = 10


def getconn():
    connector = Connector()
    conn = connector.connect(
        DB_INSTANCE, "pg8000", user=DB_USER, password=DB_PASS, db=DB_NAME
    )
    return conn


def getpool():
    return sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )


def embed_query(query):
    return model.encode(query)


def fetch_k_nearest(embedded_query, k):
    select = sqlalchemy.text(
        """
        SELECT file_name, context FROM legislation_vector_db_003
        ORDER BY embedding <-> (:embedding)
        LIMIT (:limit)
        """
    )

    pool = getpool()
    with pool.connect() as db_conn:
        result = db_conn.execute(
            select, parameters={"embedding": str(list(embedded_query)), "limit": k}
        ).fetchall()

    return [str({str(row[0]).replace(".txt", ""): str(row[1])}) for row in result]


def prompt(query, context):
    input = f"""
        \n\nHuman: 
        QUESTION:
        Here is a question about United States (US) legislation: {query}.
        DIRECTIONS:
        Soon I will provide some context and data for you to answer this question.
        ONLY answer the question if you feel you can answer it well, and if it relates to the
        given context.
        
        This context is a dictionary with legislative bill names as keys and extracted excerpts as
        values. These bills have been proposed in either the US House or Senate.

        Senate bills have an S in their file name while House bills have HR in their file name. 
        When answering, please be sure to reference this data in your answer. Please format your
        answer as follows:

        {{**Your general answer based on the given context and your existing knowledge. This should
        be about 3 sentences long and MUST provide a specific insight or thorough summary of the
        provided context. PLEASE BE DETAILED.**}}
         {{**Bill (insert relevant bill name here)**:\n
            "A direct quote from the bill"\n
            - bullet 1 that explains how this bill is related to the query\n
            - bullet 2...}}.\n

        EXAMPLE OUTPUT (DO NOT USE THIS AS ACCURATE DATA):
        **Congress prioritizes energy storage technology over climate mitigation. Several bills
        provide investment in emerging energy technologies. Most notably, HR-650 proposes to
        create a new division in the Department of Energy for research and development.**

        **Bill HR-650: Energy Storage Act**\n
            "establishment of a division of the DOE for the purposes of energy infrastructure"\n
            - Establishes a new division in the Department of Energy.\n
            - Places significant emphasis on energy infrastructure and storage technology.\n

        INSTRUCTIONS CONTINUED:
        Include as many bills as you feel are relevant. Please remain consistent with the above format
        to ensure the answer is readable with bullet points, which can be denoted by a dash (-).
        Please bold your general answer, as well as the titles of each specific bill to make them
        stand out. This can be done by surrounding the sentence with two asterisks, as shown in the
        example formatting above. IMPORTANT: do not include brackets in your answer; 
        the brackets are simply to indicate the format for you to place your response!
        When in doubt about format, please format it in a human readable way and using Markdown.
        Always include the bill name.
        Please be detailed regarding how this legislation relates
        to the question, citing parts of the context at least once for each bill you mention.
        When citing, if you notice a word that has been cut-off / is incomplete, please start the
        quote after that word, or end before that word if it is the last word. Essentially,
        you do not need to cite the entire context for that bill, you can chose what part makes sense to cite.
        Feel free to use any additional knowledge if it will improve the answer but the
        given context is the most important. Please only provide the answer, do not directly
        acknowledge my formatting requests in your answer. For instance, do not say that you
        have summarized documents provided or received context of any sort.\n
        CONTEXT: {context}\n\nAssistant:
        """

    anthropic = Anthropic(api_key=CLAUDE_KEY)
    return str(
        anthropic.completions.create(
            model="claude-instant-1.2", max_tokens_to_sample=10000, prompt=input
        ).completion
    )


@functions_framework.http
def answer_query(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    start = time.time()
    request_json = request.get_json(silent=True)
    QUERY_PARAM = "query"

    if request_json and QUERY_PARAM in request_json:
        query = request_json[QUERY_PARAM]
    else:
        return "Status 400"

    print("entry")
    embedded_query = embed_query(query)
    print("embed")
    context = fetch_k_nearest(embedded_query, K_NEAREST)
    print("got context")
    print(time.time() - start)
    answer = {"answer": prompt(query, str(context)), "context": context}
    print(time.time() - start)
    return answer
