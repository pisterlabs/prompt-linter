from fastapi import FastAPI, HTTPException
import uvicorn
from typing import Dict

import os
import sys
from dotenv import load_dotenv
load_dotenv('.env')

sys.path.append(os.getcwd().split(os.environ['PROJECT_NAME'])[0] + os.environ['PROJECT_NAME'] + '/src')
import langchain_functions

app = FastAPI()

@app.post("/generate_docstring")
def generate_docstring(request: Dict):
    try:
        # Run the LangChain function to generate the docstring
        output_code = langchain_functions.generate_docstring(request['file_contents'])
        
        # Return the code with docstring
        return {"output_code": output_code}
    
    except Exception as e:
        # Return an error if something goes wrong
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
async def root():
    return {"message": "API Running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)