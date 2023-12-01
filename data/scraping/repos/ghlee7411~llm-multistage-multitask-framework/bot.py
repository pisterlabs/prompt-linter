from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dataclasses import dataclass, field
load_dotenv(verbose=True, override=True)
del load_dotenv


def recipe_search_engine(input: str):
    """Recipe Search Engine. Ask any question and get the answer."""
    embeddings = OpenAIEmbeddings()
    db = Chroma(
        persist_directory='recipe_items_chroma', 
        embedding_function=embeddings
    )
    documents = db.search(input, search_type='similarity', k=3)
    output = ''
    for document in documents:
        content = document.page_content
        # pretty json
        content = content.replace('},', '},\n')
        output += content
    return output


app = FastAPI()

class InferenceRequest(BaseModel):
    text: str
    document_k: int = 10
    document_search_type: str = 'similarity'
    temperature: float = 0.9


@app.post("/inference/")
async def inference(request: InferenceRequest):
    try:
        # Load the recipe vector store
        print('Load the recipe vector store')
        embeddings = OpenAIEmbeddings()
        db = Chroma(
            persist_directory='recipe_items_chroma', 
            embedding_function=embeddings
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to load the recipe vector store: {e}"
        )

    try:
        # 1. Retrieve the recipe embeddings
        print('Retrieve the recipe embeddings')
        documents = db.search(
            request.text, 
            search_type=request.document_search_type, 
            k=request.document_k
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to load the recipe embeddings: {e}"
        )

    try:
        # 2. Rank the retrieved recipes by their semantic similarity to the input
        print('Rank the retrieved recipes by their semantic similarity to the input')
        from prompt import DOCUMENT_RERANKING_PROMPT_TEMPLATE
        prompt = DOCUMENT_RERANKING_PROMPT_TEMPLATE.format(
            question=request.text, 
            documents='\n'.join([d.page_content for d in documents])
        )
        rerank_llm = OpenAI(temperature=request.temperature, verbose=False)
        reranked_documents = rerank_llm(prompt)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to rank the retrieved recipes: {e}"
        )

    try:
        # 3. Summarize the top ranked recipes
        print('Summarize the top ranked recipes')
        from prompt import DOCUMENT_SUMMARIZATION_PROMPT_TEMPLATE
        prompt = DOCUMENT_SUMMARIZATION_PROMPT_TEMPLATE.format(
            question=request.text,
            documents=reranked_documents
        )
        summary_llm = OpenAI(temperature=request.temperature, verbose=False)
        summary = summary_llm(prompt)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to summarize the top ranked recipes: {e}"
        )
    
    try:
        # 4. Synthesize the key elements of the top ranked recipes
        print('Synthesize the key elements of the top ranked recipes')
        from prompt import DOCUMENT_SYNTHESIS_PROMPT_TEMPLATE
        prompt = DOCUMENT_SYNTHESIS_PROMPT_TEMPLATE.format(
            question=request.text,
            documents=reranked_documents
        )
        synthesis_llm = OpenAI(temperature=request.temperature, verbose=False)
        synthesis = synthesis_llm(prompt)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to synthesize the key elements of the top ranked recipes: {e}"
        )
    
    try:
        # 5. Generate the final answer by combining the summary and synthesis
        print('Generate the final answer by combining the summary and synthesis')
        from prompt import ANSWER_GENERATION_PROMPT_TEMPLATE
        prompt = ANSWER_GENERATION_PROMPT_TEMPLATE.format(
            question=request.text,
            documents=reranked_documents,
            summary=summary,
            key_elements=synthesis
        )
        answer_llm = OpenAI(temperature=request.temperature, verbose=False)
        answer = answer_llm(prompt)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate the final answer by combining the summary and synthesis: {e}"
        )
    
    return {
        'answer': answer,
        'reasoning': {
            'summary': summary,
            'key_elements': synthesis,
            'documents': reranked_documents
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)