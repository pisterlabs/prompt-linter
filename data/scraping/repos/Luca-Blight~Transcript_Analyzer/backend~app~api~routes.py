import asyncio
import os
import tempfile

from fastapi import APIRouter, UploadFile
from fastapi.responses import HTMLResponse
from langchain.chat_models import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv, find_dotenv
from prompts import product_prompt_template, final_product_prompt_template
from utils.document_utils import extract_text_from_pdf, split_into_chunks


_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

router = APIRouter()



@router.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>My Cool FastAPI App</title>
            <style>
                body {
                    background-color: lightblue;
                }
                h1 {
                    color: white;
                    text-align: center;
                    padding-top: 50px;
                }
            </style>
        </head>
        <body>
            <h1>Welcome to my FastAPI app!</h1>
        </body>
    </html>
    """


@router.post("/analyze")
async def analyze_document(file: UploadFile) -> dict:
    start = time.time()
    filename = file.filename
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as executor:
        if filename.endswith(".pdf"):

            extracted_text = await loop.run_in_executor(
                executor, extract_text_from_pdf, file.file
            )

            chunks = await loop.run_in_executor(
                executor, split_into_chunks, extracted_text
            )
            chat = ChatOpenAI(temperature=0.0, model="gpt-4")
            
            # run tasks in parallel
            start_chat_tasks = time.time()
            tasks = [loop.run_in_executor(executor, chat, product_prompt_template.format_messages(text=chunk)) for chunk in chunks]
            insights = await asyncio.gather(*tasks)
            print(f'Time taken for initial chunk insights: {time.time() - start_chat_tasks} seconds')

            #append insights into final product prompt
            start_agg_insights = time.time()
            agg_insights = final_product_prompt_template.format_messages(text=insights)
            final_insights = await loop.run_in_executor(executor, chat, agg_insights)
            print(f'Time taken for final insights: {time.time() - start_agg_insights} seconds')

            execution_time = time.time() - start
            print(f'Time taken: {execution_time} seconds')
            return final_insights

        elif file.endswith(".txt"):
            return "This is a text file."
        else:
            return "Error: This is not a PDF or text file."

