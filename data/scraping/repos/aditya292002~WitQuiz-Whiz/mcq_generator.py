# from typing import Annotated
from fastapi import APIRouter
from icecream import ic
from fastapi import Request, Form, FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utility.scrapper import *
from openai_handler import *
import asyncio
from fastapi.templating import Jinja2Templates
from utility.parse_mcq import parse_mcq
from utility.generate_mcq import generate

router_gen_mcq = APIRouter(
    prefix='/process',
    tags=['process']
)


class URL(BaseModel): 
    url: str


templates = Jinja2Templates(directory="templates")







@router_gen_mcq.post("/urls")
async def process_urls(request: Request, url: str = Form(...)):
    print(url)
    data = scrape_web_content(url)
    
    # for the sake of simplicity and avoid overuse of API i am considering only 3 chunks of data
    data = data[:min(len(data), 1600)]
    chunks = []
    ind = 0
    count_chunks = 0
    while(ind < len(data) and count_chunks < 3):
        chunks.append(' '.join(data[ind:ind+500][:30]))
        ind += 500
        count_chunks += 1
        
    ic(chunks)

    
    mcqs = await asyncio.gather(*(generate(chunk) for chunk in chunks))
    parsed_mcq = parse_mcq(mcqs)
    print(parsed_mcq)
    
        
    return templates.TemplateResponse("app.html", {"request": request, "mcqs": parsed_mcq})
      
    
@router_gen_mcq.post("/pdf")
async def upload_pdf(request: Request, pdf: UploadFile = File(...)):
    ic("inside upload pdf post")
    # Check if the uploaded file is a PDF
    if pdf.filename.endswith(".pdf"):
        pdf_content = await pdf.read()  # Asynchronously read the contents of the PDF file
        data = extract_pdf_content(pdf_content)

        # For simplicity, consider only 3 chunks of data
        data = data[:min(len(data), 1600)]
        chunks = [' '.join(data[i:i+500][:30]) for i in range(0, len(data), 500)][:3]
        ic(chunks)

        mcqs = await asyncio.gather(*(generate(chunk) for chunk in chunks))
        print(mcqs)

        return templates.TemplateResponse("app.html", {"request": request, "mcqs": mcqs})

    else:
        return JSONResponse(content={"error": "Please upload a PDF file"}, status_code=400)
    
@router_gen_mcq.get("/pdf")
async def upload_pdf(request: Request, pdf: UploadFile = File(...)):
    ic("inside upload pdf get")
    # Check if the uploaded file is a PDF
    if pdf.filename.endswith(".pdf"):
        data =extract_pdf_content(pdf)
        # for the sake of simplicity and avoid overuse of API i am considering only 3 chunks of data
        data = data[:min(len(data), 1600)]
        chunks = []
        ind = 0
        count_chunks = 0
        while(ind < len(data) and count_chunks < 3):
            chunks.append(' '.join(data[ind:ind+500][:30]))
            ind += 500
            count_chunks += 1
            
        ic(chunks)

        
        mcqs = await asyncio.gather(*(generate(chunk) for chunk in chunks))
        print(mcqs)
        ans = parse_mcq(mcqs)
        print("answer ----------------------------------------------", ans)
        # for mcq in mcqs:
            
        return templates.TemplateResponse("app.html", {"request": request, "mcqs": mcqs})
        
    
        
        
    else:
        return JSONResponse(content={"error": "Please upload a PDF file"}, status_code=400)
    
    
    
    
    
    
    