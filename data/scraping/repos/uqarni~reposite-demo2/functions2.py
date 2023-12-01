import openai
import os
import re
import random
from datetime import datetime, timedelta
import random
import time

#similarity search
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

def find_txt_examples(query, k=8):
    loader = TextLoader("examples.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(docs, embeddings)
    docs = db.similarity_search(query, k=k)

    examples = ""
    for doc in docs:
       examples += '\n\n' + doc.page_content
    return examples


def find_examples(query, type, k=8):
    if type == 'taylor_RAG':
        full_file = 'RAG_examples/taylor.csv'
        col1 = 'RAG_examples/taylorcol1.csv'

    elif type == 'taylorNMQR_RAG':
        full_file = 'RAG_examples/taylorNMQR.csv'
        col1 = 'RAG_examples/taylorNMQRcol1.csv'
        
    loader = CSVLoader(file_path=col1)

    data = loader.load()
    embeddings = OpenAIEmbeddings()


    db = FAISS.from_documents(data, embeddings)
    examples = ''
    docs = db.similarity_search(query, k)
    df = pd.read_csv(full_file)
    i = 1
    for doc in docs:
        input_text = doc.page_content[14:]
        try:
            output = df.loc[df['User Message'] == input_text, 'Assistant Message'].iloc[0]
        except:
            print('found error for input')

        try:
            examples += f'Example {i}: \n\nLead Email: {input_text} \n\nTaylor Response: {output} \n\n'
        except:
            continue
        i += 1
    return examples


def my_function(og, permuted):
    try:
        output = find_examples(permuted, k = 10)
        if og in output:
            return 'yes'
        else:
            return 'no'
    except:
        print('error')
        print('\n\n')
        return 'error'
    
# Read CSV
def find_in_examples_script():
    df = pd.read_csv('oct12comparison.csv')

    # Apply function to each row and store result in a new column
    df['Output'] = df.apply(lambda row: my_function(row['Assistant Reference Message'], row['Modified user message']), axis=1)

    # Write DataFrame back to CSV
    df.to_csv('oct12comparison_modified.csv', index=False)


















#generate openai response; returns messages with openai response
def ideator(messages, lead_dict_info, bot_used):
    print('message length: ' + str(len(messages)))
    prompt = messages[0]['content']
    messages = messages[1:]
    new_message = messages[-1]['content']

    #perform similarity search
    examples = find_examples(new_message, bot_used, k=4)
    examples = examples.format(**lead_dict_info)
    prompt = prompt + examples
    print('inbound message: ' + str(messages[-1]))
    print('prompt' + prompt)
    print('\n\n')
    prompt = {'role': 'system', 'content': prompt}
    messages.insert(0,prompt)
    
    for i in range(5):
      try:
        key = os.environ.get("OPENAI_API_KEY")
        openai.api_key = key
    
        result = openai.ChatCompletion.create(
          model="gpt-4",
          messages= messages,
          max_tokens = 500,
          temperature = 0
        )
        response = result["choices"][0]["message"]["content"]
        response = response.replace('\n','<br>')
        print('response:')
        print(response)
        print('\n\n')
        break
      except Exception as e: 
        error_message = f"Attempt {i + 1} failed: {e}"
        print(error_message)
        if i < 4:  # we don't want to wait after the last try
          time.sleep(5)  # wait for 5 seconds before the next attempt
  
    def split_sms(message):
        import re
  
        # Use regular expressions to split the string at ., !, or ? followed by a space or newline
        sentences = re.split('(?<=[.!?]) (?=\\S)|(?<=[.!?])\n', message.strip())
        # Strip leading and trailing whitespace from each sentence
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
  
        # Compute the cumulative length of all sentences
        cum_length = [0]
        for sentence in sentences:
            cum_length.append(cum_length[-1] + len(sentence))
      
            total_length = cum_length[-1]
  
        # Find the splitting point
        split_point = next(i for i, cum_len in enumerate(cum_length) if cum_len >= total_length / 2)
  
        # Split the sentences into two parts at the splitting point
        part1 = sentences[:split_point]
        part2 = sentences[split_point:]
  
        # Join the sentences in each part back into strings and exclude any part that is empty
        strings = []
        if part1:
            strings.append(" ".join(part1))
        if part2:
            strings.append(" ".join(part2))
      
        return strings

    split_response = [response]
    count = len(split_response)
    for section in split_response:
        section = {
           "role": "assistant", 
           "content": section
           }
        messages.append(section)

    return messages, count





def initial_text_info(selection=None):
    dictionary = {

        'NMQR': '''
        Hey {lead_first_name} -

I just saw you got a group reservation request through Reposite from {reseller_org_name}!

Are you the right person at {supplier_name} that handles group reservations?

Cheers,

Taylor
''',

        'NTM $500 Membership - New QR':'''
        Hey {lead_first_name} -

I saw that your Reposite profile just sparked some new interest! A planner {reseller_org_name}, just sent you a new quote request - they're looking for {category} suppliers in {destination}.

Based on the details, do you feel like this lead is relevant for {supplier_name}?

Cheers,
Taylor
''',

        'NTM $500 Membership - Token Change':'''
Hey {lead_first_name} -

I saw that you just used tokens to discover new group planners. It's great to see you taking active steps to expand your connections!

Are there certain types of planners that you're targeting (corporate, student groups, international groups, luxury, etc.)?

Cheers,
Taylor
''',
        'NTM $500 Membership - Quote Hotlist':'''
Hey {lead_first_name} -

I noticed that your conversation with {reseller_org_name} is off to a good start - congrats (though I don't want to jinx it)!

Are you open to receiving more quotes and group leads from other planners?

Cheers,
Taylor
''',
        'NTM $500 Membership - Booking Received': '''
Hey {lead_first_name} -

Congrats on your recent booking with {reseller_org_name}! Was everything up to your expectations?

Best,
Taylor
'''
    }
    if selection is None:
      return list(dictionary.keys())
    
    return dictionary[selection]