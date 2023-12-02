import os
import cohere
import pinecone
import requests
from tqdm import tqdm
from fastapi import FastAPI, UploadFile, HTTPException, Form
from pydantic import BaseModel
from pytube import YouTube

from assembly.file_reader import get_paragraphs_from_pdf, get_paragraphs_from_epub, enumerate_p_tags_epub
from assembly.utils import start_transcription, get_paragraphs_from_transcript, get_info_from_transcript

AZURE_WEBHOOK_ENDPOINT = 'https://assembly.ayfdhubah8c7hvg5.germanywestcentral.azurecontainer.io/assembly'

TRANSCRIPTION_EDGE_FUNCTION = 'https://ctejpbhbokuzzivioffl.functions.supabase.co/assembly-webhook'

app = FastAPI()

co = cohere.Client(api_key=os.environ['COHERE_API_KEY'])

pinecone.init(api_key=os.environ['PINECONE_TEXT_API_KEY'])
paragraphs_index = pinecone.Index('paragraphs')


transcription_id_to_video_id = {}


'''
Submit a video for transcription

This endpoint submits a video for transcription and returns the transcription id.
'''
class TranscribeRequest(BaseModel):
    video_url: str
    video_id: str

@app.post('/transcribe')
def transcribe(request: TranscribeRequest):
    print(f'Transcribing {request.video_url}...')

    # Get video info from youtube
    yt = YouTube(request.video_url)
    video_title = yt.title
    video_thumbnail = yt.thumbnail_url

    # Submit for transcription
    transcription_response = start_transcription(request.video_url, webhook=AZURE_WEBHOOK_ENDPOINT)

    response_body = {
        'video_id': request.video_id,
        'video_title': video_title,
        'video_thumbnail': video_thumbnail,
        'transcription_id': transcription_response['id'],
    }

    transcription_id_to_video_id[transcription_response['id']] = request.video_id

    return response_body


'''
Query endpoint for pinecone

This endpoint is used to make queries to the pinecone index.
'''
class QueryRequest(BaseModel):
    query: str

@app.post('/query')
def post_query(request: QueryRequest):
    # Embed query
    cohere_response = co.embed([request.query], model='large')
    query_embedding = cohere_response.embeddings[0]

    # Query pinecone
    query_results = paragraphs_index.query(
        query_embedding,
        top_k=5,
        include_metadata=True)['matches']

    # Construct response
    response_items = []
    for entry in query_results:
        response_items.append({
            'score': entry['score'],
            'resource_id': entry.metadata['resource_id'],
            'paragraph_id': entry.metadata['paragraph_id'],
            'text': entry.metadata['text'],
            'content_type': entry.metadata['content_type'],
        })

    return response_items


'''
Webhook endpoint for AssemblyAI

This endpoint is called when a transcription is completed.
It calls a Supabase edge-function to upload the transcription.
'''

class AssemblyRequest(BaseModel):
    transcript_id: str
    status: str

@app.post('/assembly')
def post_assembly(request: AssemblyRequest):
    if request.status != 'completed':
        return

    if request.transcript_id not in transcription_id_to_video_id:
        return

    video_id = transcription_id_to_video_id[request.transcript_id]
    print(f'Assembly {request.transcript_id} for video {video_id} completed!')

    # Get paragraphs from Assembly
    paragraphs_response = get_paragraphs_from_transcript(request.transcript_id) 
    paragraphs = paragraphs_response['paragraphs']
    paragraph_texts = [p['text'] for p in paragraphs]

    # Get transcript info from Assmbly
    transcript_response = get_info_from_transcript(request.transcript_id)
    summary = transcript_response['summary']

    # Get paragraph embeddings
    print('Getting paragraph embeddings...')
    cohere_response = co.embed(paragraph_texts, model='large')
    paragraph_embeddings = cohere_response.embeddings

    # Upload embeddings to pinecone
    print(f'Uploading {len(paragraph_embeddings)} embeddings to pinecone...')
    data = []
    for i, embedding in tqdm(enumerate(paragraph_embeddings)):
        data.append((
            f'{video_id}_{i}',
            embedding,
            {
                'resource_id': video_id,
                'paragraph_id': i,
                'text': paragraph_texts[i],
                'content_type': 'video'
            }
        ))

        if len(data) == 32 or i == (len(paragraph_embeddings) - 1):
            paragraphs_index.upsert(data)
            data = []

    # Call edge function to upload transcription
    print('Calling edge function...')
    supabase_headers = { 'Authorization': f'Bearer {os.environ["SUPABASE_ANON_KEY"]}' }
    supabase_body = {
        'paragraphs': paragraphs,
        'transcription_id': request.transcript_id,
        'video_id': video_id,
        'summary': summary,
    }

    supabase_response = requests.post(
        TRANSCRIPTION_EDGE_FUNCTION,
        headers=supabase_headers,
        json=supabase_body,
    ).json()
    
    print(supabase_response)

    return {'status': 'ok'}


'''
Submit a file for embedding

This endpoint creates paragraph embeddings and uploads them to pinecone.
'''

@app.post('/upload')
async def post_upload(file: UploadFile, resource_id: str = Form()):
    print(f'Received {file.filename} of type: {file.content_type}')
    if file.content_type not in ['application/pdf', 'application/epub+zip']:
        raise HTTPException(status_code=400, detail='File type not supported')

    # Temporarily save file
    paragraphs: list[tuple[str, int|str]] = []
    file_contents = await file.read()
    local_path = f'./tmp/{file.filename}'
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with open(local_path, 'wb') as f:
        f.write(file_contents)

    # Get paragraphs from file
    if file.content_type == 'application/pdf':
        paragraphs = get_paragraphs_from_pdf(local_path)
    elif file.content_type == 'application/epub+zip':
        enumerate_p_tags_epub(local_path)
        paragraphs = get_paragraphs_from_epub(local_path)

    os.remove(local_path)

    # Embed paragraphs
    paragraphs_texts, paragraphs_ids = map(list, zip(*paragraphs))
    print(f'Embedding {len(paragraphs)} paragraphs...')

    response = co.embed(paragraphs_texts, model='large')
    page_embeddings = response.embeddings
    print(f'Cohere embeddings: {len(page_embeddings)}, dim: {len(page_embeddings[0])}')

    print(f'Finished embedding, uploading to pinecone ...')

    data = []
    for i in range(len(page_embeddings)):
        data.append((
            f'{resource_id}-{i}',
            page_embeddings[i],
            {
                'resource_id': resource_id,
                'paragraph_id': paragraphs_ids[i],
                'text': paragraphs_texts[i],
                'content_type': 'pdf' if file.content_type == 'application/pdf' else 'epub'
            }
        ))

        if len(data) == 32 or i == (len(paragraphs) - 1):
            paragraphs_index.upsert(data)
            data = []

    return {'num_embeddings': len(page_embeddings), 'resource_id': resource_id}

