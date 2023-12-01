from PyPDF2 import PdfReader 
from fastapi import HTTPException
import requests as r
import openai
from app.config import CHAT_MODEL, MAX_TOKENS, MODEL_TOKEN_LIMIT, OPENAI_API_KEY
import tiktoken
import uuid
import requests
import os

openai.api_key = OPENAI_API_KEY
def url_ok(url):
    return requests.head(url, allow_redirects=True).status_code == 200
    
def is_pdf(url):
    #check if the url is valid or not
    if not url_ok(url):
        raise HTTPException(status_code=400, detail="this is not a valid url")
    
    response = requests.head(url)
    content_type = response.headers.get('Content-Type', '')
    if content_type.lower() == 'application/pdf':
        return True
    else:
        return False



def get_pdf_file_content(pdf_url: str) -> str:
    #check if the url contains a PDF file 
    if not is_pdf(pdf_url):
        raise HTTPException(status_code=400, detail="this is not a pdf file")

    response = r.get(pdf_url)

    #generate random names for pdf files
    pdf_temp_name = str(uuid.uuid4())+'.pdf'

    with open(pdf_temp_name, 'wb') as pdf_file:
        pdf_file.write(response.content)

    try:
        reader = PdfReader(pdf_temp_name)
    except:
        raise HTTPException(status_code=400, detail="this is not a valid pdf file")
    
    text = ''
    for page in reader.pages:
        text += page.extract_text() 
    os.remove(pdf_temp_name)
    return text


def openaiCompletionApi(url):
    pdfContent = get_pdf_file_content(url)
    if not pdfContent:
        raise HTTPException(status_code=400, detail="PDF does not contain any text")
    prompt = "generate a lesson plan associated with the content of the PDF below. The plan must be as the same language as the PDF language"
    
    # Initialize the tokenizer
    tokenizer = tiktoken.encoding_for_model(CHAT_MODEL)
    # Encode the text_data into token integers
    token_integers = tokenizer.encode(pdfContent)
    # Split the token integers into chunks based on max_tokens
    chunk_size = MAX_TOKENS - len(tokenizer.encode(prompt))
    chunks = [
        token_integers[i : i + chunk_size]
        for i in range(0, len(token_integers), chunk_size)
    ]
    
    # Decode token chunks back to strings
    chunks = [tokenizer.decode(chunk) for chunk in chunks]

    messages = [
        {"role": "user", "content": prompt},
        {
            "role": "user",
            "content": "To provide the context for the above prompt, I will send you PDF content in parts. When I am finished, I will tell you 'ALL PARTS SENT'. Do not answer until you have received all the parts.",
        },
    ]

    for chunk in chunks:
        messages.append({"role": "user", "content": chunk})
        # Check if total tokens exceed the model's limit
        if sum(len(tokenizer.encode(msg["content"])) for msg in messages) > MODEL_TOKEN_LIMIT:
            raise HTTPException(status_code=400, detail="PDF file is too large")


    # Add the final "ALL PARTS SENT" message
    messages.append({"role": "user", "content": "ALL PARTS SENT"})
    messages.append({"role": "user", "content": "The lesson plan must summarize and address the major topics mentioned in the input content but must in no way be a complete paraphrase of the content. The result must be a document Valid Markdown."})
    messages.append({"role": "user", "content": "Do not mention anything about timing of each part or short summarys  or anything else.Just give the lesson plan with a valid markdown"})
    response = openai.ChatCompletion.create(model=CHAT_MODEL, messages=messages)
    res = response.choices[0].message["content"].strip()
    md_temp_name = str(uuid.uuid4())+'.md'
    with open(md_temp_name, 'wb') as plan:
        plan.write(res.encode())
    return md_temp_name
    '''
    partie2 bonus : géerer des ressources annexes
    prompt2 = "give me link resources from this text"
    msg = [
        {"role": "user", "content": prompt2},
        {
            "role": "user",
            "content": res
        },
    ]
    response2 = openai.ChatCompletion.create(model=CHAT_MODEL, messages=msg)  
    final_response_2 = response2.choices[0].message["content"].strip()
    print(final_response_2)  
    '''
