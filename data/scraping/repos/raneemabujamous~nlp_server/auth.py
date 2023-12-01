from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from llama_index import GPTVectorStoreIndex, download_loader

router = APIRouter()

import openai

@router.post('/get-answer/website')
async def get_answer(request: Request):
    try:
        data = await request.json()
        question = data.get('question', '')
        url = data.get('url', '')
        openai.api_key = data.get('token', '')
        SimpleWebPageReader = download_loader("SimpleWebPageReader")
        loader = SimpleWebPageReader(html_to_text=True)
        documents = loader.load_data(urls=[url])
        index = GPTVectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        response = query_engine.query(question)

        if response:
            answer_text = response.response
        else:
            answer_text = "No answer found."

        return {"answer": answer_text}
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)


# body example
# {

# "url": "https://hu.edu.jo/", 
# "question": " كيفك ",
# "token":"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",
# }
