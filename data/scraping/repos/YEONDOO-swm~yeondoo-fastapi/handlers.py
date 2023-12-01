from fastapi import Query, HTTPException
import arxiv
import chromadb
from chromadb.utils import embedding_functions
from utils import *
import os
from prompts import *
import openai
from fastapi.responses import StreamingResponse, JSONResponse
from ports import *
from typing import Annotated
import requests
import re
import httpx
from database import *
from collections import defaultdict
import boto3

Google_API_KEY = os.environ["GOOGLE_API_KEY"]
Google_SEARCH_ENGINE_ID = os.environ["GOOGLE_SEARCH_ENGINE_ID"]

def get_papers(query : str = Query(None,description = "검색 키워드")):
    
    token_limit_exceeded = False
    papers = []
    search_query = "site:arxiv.org " + query
    url = f"https://www.googleapis.com/customsearch/v1?key={Google_API_KEY}&cx={Google_SEARCH_ENGINE_ID}&q={search_query}&start=0"
    res = requests.get(url).json()

    try:
        search_result = res.get("items")

        pattern = r'^\d+\.\d+$'

        paper_list = []

        for i in range(len(search_result)):
            paper_id = search_result[i]['link'].split('/')[-1]
            if paper_id in paper_list:
                continue
            if bool(re.match(pattern, paper_id)):
                paper_list.append(search_result[i]['link'].split('/')[-1])
                
        search = arxiv.Search(
            id_list = paper_list,
            max_results = len(paper_list),
            sort_by = arxiv.SortCriterion.Relevance,
            sort_order = arxiv.SortOrder.Descending
        )

        for result in search.results():
            paper_info={}
            paper_info["paperId"] = result.entry_id.split('/')[-1][:-2]
            paper_info["year"] = int(result.published.year)
            paper_info["title"] = result.title
            paper_info["authors"] = [author.name for author in result.authors]
            paper_info["summary"] = result.summary
            paper_info["url"] = result.entry_id
            paper_info["categories"] = result.categories
            papers.append(paper_info)
    except:
        token_limit_exceeded = True

    search = arxiv.Search(
        query = "'"+query+"'",
        max_results = 20,#50개당 1초 소요 
        sort_by = arxiv.SortCriterion.Relevance,
        sort_order = arxiv.SortOrder.Descending
    )



    for result in search.results():
        paper_info={}
        paper_info["paperId"] = result.entry_id.split('/')[-1][:-2]
        paper_info["year"] = int(result.published.year)
        paper_info["title"] = result.title
        paper_info["authors"] = [author.name for author in result.authors]
        paper_info["summary"] = result.summary
        paper_info["url"] = result.entry_id
        paper_info["categories"] = result.categories
        papers.append(paper_info)


    return  {
        "papers":papers,
        "token_limit_exceeded" : token_limit_exceeded,
    }


async def get_chat(paperId : str = Query(None,description = "논문 ID"), 
                   userPdf : bool = Query(None,description = "유저 pdf 업로드 여부"),
                   ):
    
    pattern = r"/"
    replacement = "."
    emb_paperId = re.sub(pattern, replacement, paperId)

    client = chromadb.HttpClient(host='10.0.140.252', port=port_chroma_db)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ['OPENAI_API_KEY'],
                model_name="text-embedding-ada-002"
    )
    references = None
    if userPdf:
        s3 = boto3.resource('s3')
        s3_key = paperId + '.pdf'
        doc_file_name = f"./log/{s3_key}"

        s3.meta.client.download_file(s3_bucket, s3_key, doc_file_name)

    else:
        search = arxiv.Search(
                    id_list = [paperId],
                    max_results = 1,
                    sort_by = arxiv.SortCriterion.Relevance,
                    sort_order = arxiv.SortOrder.Descending
            )
        result = next(search.results())

        prefix = "gs://arxiv-dataset/arxiv/arxiv/pdf"
        if paperId == emb_paperId:
            src_file_name = os.path.join(prefix,paperId.split('.')[0],result.entry_id.split("/")[-1]+".pdf")

            doc_file_name = os.path.join("./log/",paperId+".pdf")
        else:
            src_file_name = os.path.join(prefix,paperId.split('/')[0],result.entry_id.split("/")[-1]+".pdf")

            doc_file_name = os.path.join("./log/",emb_paperId+".pdf")

        cmd="gsutil -m cp "+src_file_name+" "+doc_file_name

        os.system(cmd)
        

        if not os.path.exists(doc_file_name):
            doc_file_name = result.download_pdf("./log/")
    
    texts = read_pdf(doc_file_name)
    references = extract_reference(texts, paperId)

    try:
        collection = client.get_collection(emb_paperId, embedding_function=openai_ef)
    
    except:
        
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(texts, disallowed_special=())

        chunks_100 = create_chunks(tokens, 100, tokenizer)
        chunks_5k = create_exact_chunks_with_overlap(tokens, 5000, 500)
        chunks_10k = create_exact_chunks_with_overlap(tokens, 10000, 500)

        text_chunks_100 = [tokenizer.decode(chunk) for chunk in chunks_100]
        text_chunks_5k = [tokenizer.decode(chunk) for chunk in chunks_5k]
        text_chunks_10k = [tokenizer.decode(chunk) for chunk in chunks_10k]


        collection = client.create_collection(name=emb_paperId,metadata = {"hnsw:space": "cosine"}, embedding_function=openai_ef)

        collection.add(
            ids = [str(i) for i in range(len(text_chunks_100))],
            documents = text_chunks_100,
        )
        for chunk in text_chunks_5k:
            context_5k = ContextCreate(text=chunk,paperId=f"{emb_paperId}_5k")
            add_data(context_5k)
        for chunk in text_chunks_10k:
            context_10k = ContextCreate(text=chunk,paperId=f"{emb_paperId}_10k")
            add_data(context_10k)

    os.remove(doc_file_name)


    return {
        "references" : references
    }

async def post_chat(data: Annotated[dict,{
                    "paperId" : str,
                    "question" : str,
                    "history" : list,
                    "extraPaperId" : str,
                    "underline" : str,
}]):
    
    id_point = defaultdict(int)
    opt = "10k"

    if data["extraPaperId"] is not None:
        opt = "5k"
    paper_context = []
    extra_context = []

    pattern = r"/"
    replacement = "."
    data['paperId'] = re.sub(pattern, replacement, data['paperId'])
    

    client = chromadb.HttpClient(host='10.0.140.252', port=8000)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ['OPENAI_API_KEY'],
                model_name="text-embedding-ada-002"
    )

    try:
        collection = client.get_collection(data['paperId'], embedding_function=openai_ef)
        
    except:
        raise HTTPException(status_code=400, detail="잘못된 요청: 임베딩 되지 않은 문서입니다.")

    if data['extraPaperId'] is not None:
        data["extraPaperId"] = re.sub(pattern, replacement, data["extraPaperId"])
        extra_id_point = defaultdict(int)
        try:
            extra_collection = client.get_collection(data['extraPaperId'], embedding_function=openai_ef)
        except:
            await get_chat(data['extraPaperId'])
            extra_collection = client.get_collection(data['extraPaperId'], embedding_function=openai_ef)

        extra_query_results = extra_collection.query(
                query_texts=data['question'],
                n_results=10,
        )
        
        for result in extra_query_results['documents'][0]:
            ctx = ContextCreate(text = result, paperId=f"{data['extraPaperId']}_{opt}")
            search_results = search_data(ctx)
            
            for search_result in search_results:
                id_integer = int(search_result.id)
                extra_id_point[id_integer] += 1

        extra_max_value = max(extra_id_point.values())  # 최대값 찾기
        extra_max_keys = [key for key, value in extra_id_point.items() if value == extra_max_value]
        r = read_data(min(extra_max_keys), f"{data['extraPaperId']}_{opt}")
        extra_context.append(r.text)

    query_results = collection.query(
        query_texts=data['question'],
        n_results=10,
    )

    for result in query_results['documents'][0]:

        ctx = ContextCreate(text = result, paperId=f"{data['paperId']}_{opt}")
        search_results = search_data(ctx)

        for search_result in search_results:
            id_integer = int(search_result.id)
            id_point[id_integer] += 1
            
    if data['history'] is not None:
        for history in data['history']:
            prev_query_results = collection.query(
                query_texts = history[0],
                n_results=5,
            )

            for result in prev_query_results['documents'][0]:

                ctx = ContextCreate(text = result, paperId=f"{data['paperId']}_{opt}")
                search_results = search_data(ctx)

                for search_result in search_results:
                    id_integer = int(search_result.id)
                    id_point[id_integer] += 1 * 0.5

    
            
    if data["underline"] is not None:
        ctx = ContextCreate(text = data["underline"], paperId=f"{data['paperId']}_{opt}")
        search_results = search_data(ctx)
        for search_result in search_results:
            id_integer = int(search_result.id)
            id_point[id_integer] += 10


    max_value = max(id_point.values())  # 최대값 찾기
    max_keys = [key for key, value in id_point.items() if value == max_value]
    r = read_data(min(max_keys), f"{data['paperId']}_{opt}")
    paper_context.append(r.text)

    
    messages = [
            {"role": "system", "content": MAIN_PROMPT}
    ]
    if data['history'] is not None:
        for history in data['history']:
            messages.append({"role": "user", "content": history[0]})
            messages.append({"role": "assistant", "content": history[1]})
            
    if data['extraPaperId'] is not None:
        if data["underline"] is not None:
            context_prompt = EXTRA_CONTEXT_WITH_UNDERLINE_PROMPT + f"\n***contex(paperid={data['paperId']}) : {paper_context}***\n" + f"\n***extra_contex(paperid={data['extraPaperId']}) : {extra_context}***\n" +f"***underline : {data['underline']}***\n" +f"***user's question : {data['question']}***"
        else:
            context_prompt = EXTRA_CONTEXT_PROMPT + f"\n***contex(paperid={data['paperId']}) : {paper_context}***\n" + f"\n***extra_contex(paperid={data['extraPaperId']}) : {extra_context}***\n" + f"***user's question : {data['question']}***"
    else:
        if data["underline"] is not None:
            context_prompt = CONTEXT_WITH_UNDERLINE_PROMPT + f"\n***contex(paperid={data['paperId']}) : {paper_context}***\n" +f"***underline : {data['underline']}***\n"+ f"***user's question : {data['question']}***"
        else:
            context_prompt = CONTEXT_PROMPT + f"\n***contex(paperid={data['paperId']}) : {paper_context}***\n" + f"***user's question : {data['question']}***"
    
    messages.append({"role": "user","content": context_prompt})

    
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens = 1000,
            stream = True,
        )

    except openai.error.Timeout as e:
        # 응답 대기 시간 초과 에러인 경우 408 응답을 반환합니다.
        
        return HTTPException(status_code=408, detail="Request Timeout")

    def generate_chunks():
        for chunk in response:
            try:
                yield chunk["choices"][0]["delta"].content + "\n"
            except:
                yield "\n"
                
    return StreamingResponse(
        content=generate_chunks(),
        media_type="text/plain"
    )


async def post_coordinates(data: Annotated[dict,{
                    "key" : str,
                    "coordinates" : list,
}]):

    # 대상 서버의 IP 주소와 포트 설정
    target_host = "10.0.129.165"
    target_port = 8080

    # 대상 서버의 URL 생성
    target_url = f"http://{target_host}:{target_port}/api/coordinates?key={data['key']}"

    # httpx를 사용하여 POST 요청 보내기
    async with httpx.AsyncClient() as client:
        payload = {"coordinates": data['coordinates']}  # 요청 데이터 준비
        response = await client.post(target_url, json=payload)

    # 응답 처리
    status_code = response.status_code
    response_data = response.json()

    return {"status_code": status_code, "response_data": response_data}



async def test(paperId : str = Query(None,description = "논문 ID"), ):
    if paperId == "be.test":

        return JSONResponse(
                status_code=500,
                content={"message": "Internal Server Error"}
        )
    
    return {
        "ok" : "ok"
    }