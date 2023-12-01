import os
import json
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import main
import pinecone
import openai
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, parse_obj_as
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from typing import Optional


main.load_dotenv()

class Query(BaseModel):
    user_input: str
    user_id: str

# Prepare augmented query
openai.api_key=os.environ['OPENAI_API_KEY']
pinecone.init(api_key=os.environ['PINECONE_API_KEY'], enviroment=os.environ['PINECONE_ENVIRONMENT'])
pinecone.whoami()
index_name = 'academyzd'
index = pinecone.Index(index_name)

embed_model = "text-embedding-ada-002"

primer = """

You are Samantha, a highly intelligent and helpful virtual assistant designed to support Ledger, a French cryptocurrency company led by CEO Pascal Gauthier. Your primary responsibility is to assist Ledger users by providing accurate answers to their questions.

Users may ask about various Ledger products, including the Ledger Nano S (no battery, low storage), Nano X (Bluetooth, large storage, has a battery), Nano S Plus (large storage, no Bluetooth, no battery), Ledger Stax, and Ledger Live.
The official Ledger store is located at https://shop.ledger.com/. For authorized resellers, please visit https://www.ledger.com/reseller/. Do not modify or share any other links for these purposes.

When users inquire about tokens, crypto or coins supported in Ledger Live, it is crucial to strictly recommend checking the Crypto Asset List link to verify support: https://support.ledger.com/hc/en-us/articles/10479755500573?docs=true/. Do NOT provide any other links to the list.

VERY IMPORTANT:

- If the query is unclear or not about Ledger products, disregard the CONTEXT and invite any Ledger-related questions using this exact response: "I'm sorry, I didn't quite understand your question. Could you please provide more details or rephrase it? Remember, I'm here to help with any Ledger-related inquiries."
- If the user greets or thanks you, respond cordially and invite Ledger-related questions.
- When responding to a question, include a maximum of two URL links from the provided CONTEXT, choose the most relevant.
- If a URL link provided in the CONTEXT is irrelevant to the USER QUESTION, you should avoid sharing it.
- Always present URLs as plain text, never use markdown formatting.
- If a user reports being victim of a scam or unauthorized crypto transactions, empathetically acknowledge their situation, promptly connect them with a live agent, and share this link for additional help: https://support.ledger.com/hc/en-us/articles/7624842382621-Loss-of-funds?support=true.
- If a user needs to reset their device, they must always ensure they have their recovery phrase on hand before proceeding with the reset.
- Updating or downloading Ledger Live must always be done via this link: https://www.ledger.com/ledger-live
- If asked about Ledger Stax, inform the user it's not yet released, but pre-orderers will be notified via email when ready to ship. Share this link for more details: https://support.ledger.com/hc/en-us/articles/7914685928221-Ledger-Stax-FAQs.
- The Ledger Recover service is not available just yet. When it does launch, keep in mind that it will be entirely optional. Even if you update your device firmware, it will NOT automatically activate the Recover service. Learn more: https://support.ledger.com/hc/en-us/articles/9579368109597-Ledger-Recover-FAQs
- If you see the error "Something went wrong - Please check that your hardware wallet is set up with the recovery phrase or passphrase associated to the selected account", it's likely your Ledger's recovery phrase doesn't match the account you're trying to access.

Begin!

"""

# #####################################################


def get_user_id(request: Request):
    try:
        body = parse_obj_as(Query, request.json())
        user_id = body.user_id
        return user_id
    except Exception as e:
        return get_remote_address(request)


# Define FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="./static/BBALP00A.TTF")

# Define limiter
limiter = Limiter(key_func=get_user_id)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Too many requests, please try again in an hour."},
    )


# Initialize user state
user_states = {}

# Define FastAPI endpoints

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/_health")
async def health_check():
    return {"status": "OK"}

@app.post('/gpt')
@limiter.limit("50/hour")
def react_description(query: Query, request: Request):
    
    global last_response
    user_id = query.user_id
    if user_id not in user_states:
        user_states[user_id] = None
    last_response = user_states[user_id]
    try:
        res_embed = openai.Embedding.create(
            input=[query.user_input],
            engine=embed_model
        )

        xq = res_embed['data'][0]['embedding']

        res_query = index.query(xq, top_k=5, include_metadata=True)
        print(res_query)

        contexts = [(item['metadata']['text'] + "\nSource: " + item['metadata'].get('source', 'N/A')) for item in res_query['matches'] if item['score'] > 0.78]

        prev_response_line = f"Samantha: {last_response}\n" if last_response else ""

        augmented_query = "CONTEXT: " + "\n\n-----\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n"+ "CHAT HISTORY: \n" + prev_response_line + "User:" + query.user_input + "\n" + "Samantha: "

    
        print(augmented_query)

        res = openai.ChatCompletion.create(
            temperature=0.0,
            model='gpt-4',
            messages=[
                {"role": "system", "content": primer},
                {"role": "user", "content": augmented_query}
            ]
        )
        response = res['choices'][0]['message']['content']

        # Save the response to the global variable
        last_response = response

        # Save the response to a thread
        user_states[user_id] = response
        print(user_states)
        print(response)
        return {'output': response}
        
    except ValueError as e:
        print(e)
        raise HTTPException(status_code=400, detail="Invalid input")


############### START COMMAND ##########

#   uvicorn bot:app --reload --port 8008
#   uvicorn memory_api_bot:app --port 80 --host 0.0.0.0
