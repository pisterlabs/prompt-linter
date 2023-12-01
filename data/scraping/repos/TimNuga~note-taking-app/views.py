from rest_framework.views import APIView
from rest_framework import generics
from .models import Note
from .serializers import NoteSerializer
from rest_framework.response import Response
from rest_framework import status
from rest_framework.exceptions import ValidationError
from django.db import DatabaseError
from django.http import Http404
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import logging
import dotenv
import os

dotenv.load_dotenv()
logger = logging.getLogger(__name__)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class NoteList(generics.ListCreateAPIView):
    queryset = Note.objects.all()
    serializer_class = NoteSerializer

    def get(self, request, *args, **kwargs):
        try:
            return super().get(request, *args, **kwargs)
        except ValidationError as e:
            return Response(e.detail, status=status.HTTP_400_BAD_REQUEST)

    def perform_create(self, serializer):
        try:
            serializer.save()
        except DatabaseError as e:
            logger.error(f"Database error occurred: {e}")
            return Response({'error': 'Failed to save note'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class NoteDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Note.objects.all()
    serializer_class = NoteSerializer

    def retrieve(self, request, *args, **kwargs):
        try:
            return super().retrieve(request, *args, **kwargs)
        except Http404:
            return Response({'error': 'Note not found'}, status=status.HTTP_404_NOT_FOUND)
        
class SummarizeNoteView(APIView):
    def get(self, request, pk):
        try:    
            note = Note.objects.get(pk=pk)
            prompt_template = "Summarize the following note:\n\n {text} \n Summary: " 
            prompt = PromptTemplate(input_variables=["text"],template=prompt_template)
            
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", api_key=OPENAI_API_KEY)
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
            
            document = Document(page_content=note.content)

            summary = stuff_chain.run({"input_documents": [document]})

            return Response({"summary":summary})
        except Note.DoesNotExist:
            return Response({"error": "Note not found"}, status= status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)