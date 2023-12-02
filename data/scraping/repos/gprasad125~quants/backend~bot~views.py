import os
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from bot.models import Question, SourceDocument
from quants.qa import process_query
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

@api_view(['GET', 'POST'])
def upload_file(request):
    '''
    Maps to /documents. API for uploading user documents
    '''
    if request.method == 'GET':

        docs = SourceDocument.objects.all()
        res = [doc.name for doc in docs]
        return Response(res, status=status.HTTP_200_OK)
    
    elif request.method == 'POST':

        files = request.data.getlist('files')
        if not files or len(files) < 1:
            return Response('No files received', status=status.HTTP_400_BAD_REQUEST)

        data = []
        sources = []
        for file in files:
            ext = os.path.splitext(file.name)[1]
            if ext.lower() in ['.md', '.txt']:
                data.append(str(file.read(), encoding='utf-8', errors='ignore'))
                sources.append(file.name)
        text_splitter = CharacterTextSplitter(chunk_size=1500, separator='\n')
        docs = []
        metadatas = []
        for i, d in enumerate(data):
            splits = text_splitter.split_text(d)
            docs.extend(splits)
            metadatas.extend([{'source': sources[i]}] * len(splits))

        embeddings = embedding.embed_documents(docs)

        SourceDocument.objects.all().delete()

        for i in range(len(embeddings)):
            SourceDocument.objects.create(
                name=metadatas[i]['source'],
                content=docs[i],
                embedding=embeddings[i]
            )
        return Response('files ingested! ready to be queried', status=status.HTTP_200_OK)


@api_view(['GET'])
def get_file(request, file_id):
    '''
    Maps to /documents/file_id/. API for getting user documents.
    '''
    try:
        doc = SourceDocument.objects.get(id=file_id)
    except SourceDocument.DoesNotExist:
        return Response('file not found!', status=status.HTTP_404_NOT_FOUND)
    
    res = {
        'name': doc.name,
        'content': doc.content
    }
    return Response(res, status=status.HTTP_200_OK)


@api_view(['POST'])
def query(request):
    '''
    Maps to query/. API for querying document database. 
    '''
    if request.method != 'POST':
        return
    
    # add new Question to database
    text = request.data.get('question', 'What is Django?')
    add = Question(text=text)
    add.save()

    query = SourceDocument.objects.all()

    # run qa langchain
    validated_answer, _, snippets = process_query(query, text)

    formatted_sources = []
    for snippet in snippets:
        doc = SourceDocument.objects.filter(id=snippet.uuid).first()
        if doc is not None:
            extract = snippet.content
            formatted_sources.append({
                'name': doc.name,
                'id': doc.id,
                'extract': extract
            })

    formatted = {
        'question': text,
        'answer': validated_answer,
        'sources': formatted_sources
    }

    # return value
    return Response(formatted, status=status.HTTP_200_OK)
