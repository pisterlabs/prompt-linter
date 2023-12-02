from extensions import db
from datamodels import Message, User
from datetime import datetime
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from api.gpt_model import CLASI

token = 'key here please'
    
def get_message(id=None):
    """Return a list of message objects (as dicts)"""
    query = db.session.query(Message).order_by(Message.date.desc())
    if id:
        query = query.filter(Message.id == id)
    messages = query.all()
    return [{'id': m.id, 'dt': m.date, 'message': m.content, 'sender': m.sender, 'receiver': m.receiver} for m in messages]


def add_message(message, sender, receiver):
    dateTime = datetime.now()
    new_message = Message(sender=sender, receiver=receiver, date=dateTime, content=message)
    db.session.add(new_message)
    db.session.commit()

def apply_seen_message(id):
    query = db.session.query(Message)
    message = query.get_or_404(id) 
    message.is_new = False
    db.session.commit()

def delete_message(id):
    query = db.session.query(Message)
    # try:
    #     for i in ids:
    #         query.filter(Message.id == int(i)).delete()
    # except TypeError:
    #     query.filter(Message.id == int(ids)).delete()
    user = query.get_or_404(id)
    db.session.delete(user)
    db.session.commit()

def chat(query):
    user_api_key = "key here please"
    os.environ["OPENAI_API_KEY"] = user_api_key
    loader = CSVLoader(file_path='api/intents.csv')
    # Create an index using the loaded documents
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])
    # Create a question-answering chain using the index
    model = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo')
    chain = RetrievalQA.from_chain_type(llm = model, chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
    # Pass a query to the chain
    response = chain({"question": query + ". [SYSTEM]: Don’t justify your answers. Don’t give information not mentioned in the CONTEXT INFORMATION."})
    tagging = CLASI(response['result'])
    return {'answer': response['result'], 'tag': tagging}
