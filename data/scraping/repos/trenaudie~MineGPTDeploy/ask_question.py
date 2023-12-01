

from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
import openai
import tiktoken



def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")
  
def makeMessages_fromPinecone_and_history(question : str, docs_metadata:list, chat_history : list):
  #creates a messages object from the various docs metadata
  scientific_context_prompt = "Vous êtes assistant de recherche scientifique. Vous répondez en utilisant SEULEMENT les documents fournis. Ajoutez aux termes scientifiques les délimiteurs LaTex `$`avant et `$` après. "
  non_scientific_context_prompt = "You are a helpful, reading assistant for the elderly. You answer their questions using their documents as context, AND your own knowledge of the world. Your response must be be in two sub-200 word paragraphs, one where you use your own knowledge and one where you use the documents."
  is_scientific = any(docmeta['metadata']['scientific'] for docmeta in docs_metadata)
  if is_scientific:
    context_prompt = scientific_context_prompt
  else:
    context_prompt = non_scientific_context_prompt
  messages = [{"role": "system", "content": context_prompt}, *chat_history]
  userdict = {'role': 'user'}
  documents = ""
  for docmeta in docs_metadata:
    documents += docmeta['metadata']['text']
  userdict['content'] = f"DOCUMENTS {documents} \n QUESTION: {question} \n"
  messages.append(userdict)
  numtokens = num_tokens_from_messages(messages)
  return messages

def ask_question(question: str, vectorstore: Pinecone,  chat_history: list[dict], user_id: str = None) -> dict:
    """
    Use a question and the chat history to return the answer and the source documents.

    Updates chat history with the question and answer.

    Returns:
    dict with keys "answer", "sources"
    - answer: str
    - sources: list of dicts
        - filename: str
        - text: str
        - page: str (not yet implemented)
        - etc.
    :param question: The question to be answered.
    :param vectorstore: A Pinecone vectorstore object.
    :param chain: A ConversationalRetrievalChain object.
    :param chat_history: A list of dictionaries representing the chat history.
    :return: A dictionary containing the answer and the source documents.
    """
    #chat_history looks like below
    #must add the context_prompt and question to the chat history to create messages

    full_filter = {'$or': [{'user_id': 0}, {'user_id': user_id}] }
    print('asking question', question, 'with user_id', user_id, 'and filter', full_filter)
    docs_metadata =  vectorstore._index.query(OpenAIEmbeddings().embed_query(question), top_k = 5, filter = full_filter, include_metadata=True)['matches']
    print("docs_metadata[0]", docs_metadata[0])
    messages = makeMessages_fromPinecone_and_history(question, docs_metadata, chat_history)
    try:
      response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages = messages,
      temperature=0.5,
      )
      answer = response['choices'][0]['message']['content']
    except openai.error.RateLimitError:
      answer = 'Sorry, OpenAI is overloaded. Here are some relevant documents.'
    sources = []
    for docmeta in docs_metadata:
        sources.append(
            {'filename': docmeta.metadata['source'], 'text': docmeta.metadata['text'], 'page_number': docmeta.metadata['page_number']})
    return {"answer": answer, "sources": sources}