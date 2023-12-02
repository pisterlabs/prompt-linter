import sys
import os
import os.path
import base64
import pickle
import cmd
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.api_core.exceptions import BadRequest
import dateutil.parser as parser
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent

MAX_MESSAGES = 100
MODEL_NAME = "text-embedding-ada-002" # Name of the model used to generate text embeddings
MAX_TOKENS = 120000 # Maximum number of tokens per chunk
CHUNK_OVERLAP = 0 # Overlap between chunks
COLLECTION_NAME = "email-index"
METRIC = "cosine"
GPT_MODEL = 'gpt-4-1106-preview'

metadata_field_info = [
    AttributeInfo(name='id', description='Message ID', type='string', is_primary_key=True),
    AttributeInfo(name='subject', description='Email subject', type='string', is_primary_key=False),
    AttributeInfo(name='from', description='Email sender', type='string', is_primary_key=False),
    AttributeInfo(name='to', description='Email recipient', type='string', is_primary_key=False),
    AttributeInfo(name='date', description='Email receipt date and time', type='string', is_primary_key=False)
]

def chunk_text(text):
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_TOKENS,
                                                   chunk_overlap=CHUNK_OVERLAP)

    # Split the tokens into chunks of MAX_TOKENS tokens
    chunks = text_splitter.split_text(text)

    # Return the chunks
    return chunks

def vectorstore_setup():
    """Load stored email index from file"""
    
    docs = pickle.load(open('email_index.pkl', 'rb'))
    vectorstore = Qdrant.from_documents(docs, embedding=OpenAIEmbeddings(), location=":memory:", collection_name=COLLECTION_NAME)
    print("Vectorstore created from emails")
    return vectorstore

def get_gmail_credentials():
    """Get Gmail credentials from credentials.json file or token.pickle file"""
    
    # If modifying these SCOPES, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Load credentials from credentials.json file
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=52102)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds

def parse_date(date_string):
    try:
        date = parser.parse(date_string)
    except ValueError:
        try:
            cleaned_date_string = re.sub(r'\.\d+_\d+$', '', date_string)
            date = parser.parse(cleaned_date_string)
        except ValueError:
            date = None
    return date

# Function to decode the message part
def decode_part(part):
    if 'body' in part.keys():
        data = part['body']['data']
    else:
        return None
    data = data.replace('-', '+').replace('_', '/')
    decoded_bytes = base64.urlsafe_b64decode(data)
    return decoded_bytes.decode('utf-8')

# Function to find the desired message part
def find_part(parts, mime_type):
    for part in parts:
        if part['mimeType'] == mime_type:
            return part
    return None

message_count = 0 # Global variable to keep track of number of messages processed

def index_gmail():
    vectorstore_path = os.getenv('VECTORSTORE_PATH')
    if not vectorstore_path:
        sys.exit("VECTORSTORE_PATH environment variable is not set")
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY environment variable is not set")
    
    creds = get_gmail_credentials()
    docs = []

    try:
        def process_email(msg):
            """Process email data and add to index"""
            global message_count
            email_data = msg['payload']['headers']

            subject = ''
            to_name = ''
            for values in email_data: 
                name = values['name']
                if name == 'From':
                    from_name = values['value']
                if name == 'To':
                    to_name = values['value']
                if name == 'Subject':
                    subject = values['value']
                if name == 'Date':
                    date_value = values['value']
                    datetime_object = parse_date(date_value)
            
            try:
                data = None
                payload = msg['payload']
                if 'parts' in payload and len(payload['parts']) > 0:
                    part = find_part(payload['parts'], 'text/plain')
                    if part:
                        data = decode_part(part)
                    else:
                        part = find_part(payload['parts'], 'text/html')
                        if part:
                            data = decode_part(part)
                if not data:
                    raise ValueError(f"Couldn't find body in message {msg['id']}")
                
                # Embed email data an add to index
                chunks = chunk_text(data)
                docs.extend([Document(page_content=chunk,
                                      metadata={'id': msg['id'], 'subject': subject, 
                                                'from': from_name, 'to': to_name, 
                                                'date': datetime_object}) for chunk in chunks])
                pickle.dump(docs, open('email_index.pkl', 'wb'))
                message_count += 1

            except Exception as e:
                print(f"\nError while processing email {msg['id']}: {e}")

        # Define a function to get all messages recursively
        def get_all_emails(gmail, query):
            messages = []
            page_token=None
            
            while True:
                try:
                    result = gmail.users().messages().list(q=query, 
                                                        userId='me', 
                                                        maxResults=MAX_MESSAGES, 
                                                        pageToken=page_token).execute()
                    messages.extend( result.get('messages', []) )
                    page_token = result.get('nextPageToken', None)
                    if (not page_token) or len(messages) >= MAX_MESSAGES:
                        break
                except HttpError as error:
                    print(f"An error occurred: {error}")
                    break
            return messages

        gmail = build('gmail', 'v1', credentials=creds)
        # A query to retrieve all emails, including archived ones
        query = "in:all"
        emails = get_all_emails(gmail, query)

        # Process and print the result
        for email in tqdm(emails, desc='Processing emails', file=sys.stdout):
            msg = gmail.users().messages().get(id=email.get('id'), userId='me', format='full').execute()
            process_email(msg)
        
        pickle.dump(docs, open('email_index.pkl', 'wb'))
        print(f"Successfully added {message_count} emails to index.")

    except Exception as error:
        print(f'An error occurred: {error}')
        raise error

def ask(query):
    openai_api_key=os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    vectorstore = vectorstore_setup()

    llm = ChatOpenAI(openai_api_key=openai_api_key, 
                     model_name=GPT_MODEL,
                     temperature=0.0)
                                            

    # Answer question using LLM and email content
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                     chain_type="stuff",
                                     retriever=vectorstore.as_retriever())
    result = qa.run(query)
    print(f'\n{result}')

def chat():
    openai_api_key=os.getenv('OPENAI_API_KEY')
    if openai_api_key is None:
        sys.exit("OPENAI_API_KEY environment variable is not set")
    vectorstore = vectorstore_setup()
    llm = ChatOpenAI(openai_api_key=openai_api_key, 
                     model_name=GPT_MODEL,
                     temperature=0.0)
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k = 10,
        return_messages=True)
    
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                     chain_type="refine",
                                     retriever=vectorstore.as_retriever())

    tools = [
        Tool(
            name = 'Email Index',
            func = qa.run,
            description=('useful to answer questions about emails and messages')
        )
    ]

    agent = initialize_agent(
        agent = 'chat-conversational-react-description',
        tools = tools,
        llm = llm,
        verbose = True,
        max_iterations = 5,
        early_stopping_method = 'generate',
        memory = conversational_memory
    )

    class InteractiveShell(cmd.Cmd):
        intro = 'Welcome to the Gmail Chat shell. Type help or ? to list commands.\n'
        prompt = '(Gmail Chat) '

        def do_quit(self, arg):
            "Exit the shell."
            print('Goodbye.')
            return True
        
        def emptyline(self):
            pass

        def default(self, arg):
            "Ask a question."
            try:
                result = agent.run(arg)
                print(f'\n{result}')
            except Exception as e:
                print(e)

    InteractiveShell().cmdloop()

def usage():
    sys.exit("""
    OPENAI_API_KEY, and VECTORSTORE_PATH environment variables must be set.
    
    Usage: gmail_chat index | ask <query> | chat
    """)

def main():
    if len(sys.argv) < 2:
        usage()

    if sys.argv[1] == 'index':
        index_gmail()
    elif sys.argv[1] == 'ask':
        if len(sys.argv) < 3:
            usage()
        ask(sys.argv[2])
    elif sys.argv[1] == 'chat':
        chat()
    else:
        usage()

if __name__ == '__main__':
    main()
