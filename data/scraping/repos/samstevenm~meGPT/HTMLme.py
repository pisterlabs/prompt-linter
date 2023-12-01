#Begin HTMLme.py v0.0.1
import time
import openai
import argparse, logging, os, sqlite3, uuid
import math
from bottle import route, run, request, post, template
from contextlib import closing
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

WORKING_NAME = "HTMLme.py"

ROLE_SYSTEM_CONTENT = """
You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. If you think there might not be a correct answer, you say so.

Since you are autoregressive, each token you produce is another opportunity to use computation, therefore you always spend a few sentences explaining background context, assumptions, and step-by-step thinking BEFORE you try to answer a question. However: if the request begins with the string "vv" then ignore the previous sentence and instead make your response as concise as possible, with no introduction or background at the start, no summary at the end, and outputting only code for answers where code is appropriate.

Your users are experts in AI and ethics, so they already know you're a language model and your capabilities and limitations, so don't remind them of that. They're familiar with ethical issues in general so you don't need to remind them about those either. Don't be verbose in your answers, but do provide details and examples where it might help the explanation. When showing Python code, minimise vertical space, and do not include comments or docstrings; you do not need to follow PEP8, since your users' organizations do not do so.
"""

if not openai.api_key:
    raise ValueError("API key not loaded from .env file. Ensure .env exists and contains OPENAI_API_KEY.")

logging.basicConfig(level=logging.DEBUG)

def initialize_database():
    with closing(sqlite3.connect('interactions.db')) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                context TEXT,
                question TEXT,
                answer TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        ''')
        conn.commit()

def like_dislike_interaction(interaction_id, feedback):
    with closing(sqlite3.connect('interactions.db')) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE interactions SET feedback=? WHERE id=?", (feedback, interaction_id,)
        )
        conn.commit()

def start_session(existing_session_id=None):
    with closing(sqlite3.connect('interactions.db')) as conn:
        cursor = conn.cursor()
        if existing_session_id:
            cursor.execute('SELECT id FROM sessions WHERE id=?', (existing_session_id,))
            if not cursor.fetchone():
                cursor.execute('INSERT INTO sessions (id) VALUES (?)', (existing_session_id,))
                conn.commit()
            logging.debug("----\nArguments: session_id=" + existing_session_id)
            return existing_session_id
        session_id = str(uuid.uuid4())
        cursor.execute('INSERT INTO sessions (id) VALUES (?)', (session_id,))
        conn.commit()
        return session_id

def store_interaction(session_id, context, question, answer):
    with closing(sqlite3.connect('interactions.db')) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO interactions (session_id, context, question, answer) VALUES (?, ?, ?, ?)",
            (session_id, context, question, answer)
        )
        conn.commit()
        return cursor.lastrowid

def fetch_context_from_session(session_id):
    with closing(sqlite3.connect('interactions.db')) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT context, question, answer FROM interactions WHERE session_id=?', (session_id,))
        interactions = cursor.fetchall()
        return "\n---\n".join(f"Context: {interaction[0]}\nQuestion Asked: {interaction[1]}\nAnswer: {interaction[2]}" for interaction in interactions)


def fetch_context_from_db(row_ids):
    with closing(sqlite3.connect('interactions.db')) as conn:
        cursor = conn.cursor()
        context = ""
        for row_id in row_ids:
            cursor.execute("SELECT context, question, answer FROM interactions WHERE id=?", (row_id,))
            result = cursor.fetchone()
            if result:
                context += f"\nContext: {result[0]}\nQuestion Asked: {result[1]}\nAnswer: {result[2]}\n---"
        return context


def fetch_context_from_file(context_file):
    with open(context_file, 'r') as file:
        return "\n" + file.read()

def ask_gpt(context, question, version):
    model_name = f"gpt-{version}"
    start_time = time.time()
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. {ROLE_SYSTEM_CONTENT}"},
                {"role": "user", "content": f"{context} {question}"}
            ]
        )
        end_time = time.time()
        elapsed_time_in_seconds = end_time - start_time
        cost_per_token_in_cents = 0.006  # example; please update it with the actual cost per token
        prompt_cost = math.ceil(response['usage']['total_tokens'] * cost_per_token_in_cents)
        logging.debug(f"----\n----\nTime Taken: {elapsed_time_in_seconds}s\n----\n----\nCost of Interaction: {prompt_cost} cents")
        return response.choices[0].message['content'].strip()
    except openai.error.OpenAIError as e:
        end_time = time.time()
        elapsed_time_in_seconds = end_time - start_time
        logging.debug(f"----\n----\nTime Taken: {elapsed_time_in_seconds}s")
        return f"Error: {e}"


@route('/')
def ask_form():
    return f"""
    <h1>{WORKING_NAME}</h1>
    <form method="POST" action="/ui/ask">
      Context:<br>
      <textarea name="context" rows="4" cols="50"></textarea><br>
      Question:<br>
      <input type="text" name="question"><br>
      SQL Rows:<br>
      <input type="text" name="sql_rows" size="60"><br>
      GPT Version:<br>
      <input type="radio" name="version" value="3.5-turbo"> 3.5-turbo<br>
      <input type="radio" name="version" value="4" checked> 4<br>
      Session ID (Optional):<br>
      <input type="text" name="session_id" size="60"><br>
      <input type="submit" value="Submit">
    </form>"""

@post('/ui/ask')
def ask_ui_server():
    context = request.forms.get('context')
    question = request.forms.get('question')
    sql_rows = request.forms.get('sql_rows')  # Get SQL row IDs from form
    version = request.forms.get('version')
    session_id = request.forms.get('session_id')

    if not session_id:
        session_id = start_session() 
    else:
        start_session(session_id)

    # Fetch context from previous interactions in the session, if exists
    context_from_session = fetch_context_from_session(session_id)
    
    context_from_db = fetch_context_from_db([int(x.strip()) for x in sql_rows.split(',')] if sql_rows else [])
    context += context_from_db + context_from_session  # Add context from session to context

    if context and question:
        answer = ask_gpt(context, question, version)
        store_interaction(session_id, context, question, answer)
        context = fetch_context_from_session(session_id)
        return template("<h1>Chat History</h1>{{scenario}}<h2>Answer</h2>{{outcome}}", scenario=context, outcome=answer)
    else:
        return template("<p>Error: Invalid input</p>")


@route('/ask', method='POST')
def ask_server():
    if not all(key in request.json for key in ('context', 'question')):
        return {"error": "Invalid input, 'context' and 'question' keys are required"}   
    context = request.json.get('context', "")
    question = request.json.get('question', "")
    session_id = request.json.get('session_id', "")
    context_file = request.json.get('context_file', "")
    sql_rows = request.json.get('sql_rows', "")
    version = request.json.get('version', "4")

    if not session_id:
        session_id = start_session() 
    else:
        start_session(session_id) 
    
    context_from_db = fetch_context_from_db([int(x.strip()) for x in sql_rows.split(',')] if sql_rows else [])
    context_from_file = fetch_context_from_file(context_file) if context_file else ""
    context += context_from_db + context_from_file

    if context and question:
        answer = ask_gpt(context, question, version)
        store_interaction(session_id, context, question, answer)
        return {"answer": answer, "session_id": session_id}
    else:
        return {"error": "Invalid input"}

def setup_argparse():
    parser = argparse.ArgumentParser(description='Interact with GPT.')
    parser.add_argument('--context', type=str, help='Context for the question')
    parser.add_argument('--question', type=str, help='Question to ask GPT')
    parser.add_argument('--context_file', type=str, help='File containing the context')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--version', type=str, choices=['3.5-turbo', '4'], default='4', help='GPT version')
    parser.add_argument('--sql_rows', type=str, help='Comma-separated row IDs for context fetching')
    parser.add_argument('--session_id', type=str, help='Session ID for continuity in interactions')
    parser.add_argument('--web_mode', action='store_true', help='Web mode.')  # add web mode flag 
    return parser.parse_args()


def main_cli():
    """
    This function is the main entry point of the program. It initializes the database, starts a session, and fetches context
    from various sources. It then prompts the user for a question in interactive mode or uses the provided question in
    non-interactive mode. It passes the aggregated context and the question to the ask_gpt function to generate an answer.
    The answer is printed to the console and stored in the database along with the session ID, context, and question.
    """
    args = setup_argparse()

    initialize_database() # Initialize DB at the start
    session_id = start_session(args.session_id if 'session_id' in args else None)
    
    context_from_session = fetch_context_from_session(session_id)
    context_from_db = fetch_context_from_db([int(x.strip()) for x in args.sql_rows.split(',')] if args.sql_rows else [])
    context_from_file = fetch_context_from_file(args.context_file) if args.context_file else ""
    context_direct = "\n" + args.context if args.context else ""

    aggregated_context = f"{context_from_session}\n{context_from_db}\n{context_from_file}\n{context_direct}".strip()

    if args.interactive:
        while True:
            question = input("Enter your question (or 'exit' to quit): ")
            if question.lower() == 'exit':
                break
            answer = ask_gpt(aggregated_context, question, args.version)
            print(answer)
            store_interaction(session_id, aggregated_context, question, answer)
            aggregated_context += f"\nQuestion Asked: {question}\nAnswer: {answer}"
    elif args.question:
        answer = ask_gpt(aggregated_context, args.question, args.version)
        print(answer)
        store_interaction(session_id, aggregated_context, args.question, answer)
    else:
        logging.error("Please provide a question or use --interactive mode.")
    
def main_web():
    run(host='localhost', port=8080)

if __name__ == "__main__":
    args = setup_argparse()    
    initialize_database()  
    # Ensure a newline is added before every log message
    logging.basicConfig(format='\n%(levelname)s:%(message)s', level=logging.DEBUG)
    if 'web_mode' in args and args.web_mode:
        main_web()
    else:
        main_cli()

#End HTMLme.py HTMLme.py v0.0.1


