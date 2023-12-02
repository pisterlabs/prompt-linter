from django.shortcuts import render
from rest_framework import viewsets, mixins
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Faiss_data
from .serializers import FaissDataSerializer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import *
from sentence_transformers import SentenceTransformer
from apps.Lin_FAISS.MyFaiss import *
from django.http import JsonResponse
import os

class FaissDataViewSet(viewsets.GenericViewSet, mixins.CreateModelMixin):
    queryset = Faiss_data.objects.all()
    serializer_class = FaissDataSerializer

    # @action(detail=False, methods=['post','get'])
    # def upload_web_url(self, request):
    #     web_url = request.data.get('web_url')
    #     categories = request.data.get('categories')
        
    #     if web_url:
    #         # 你的 LinFaiss 实例化
    #         lin_faiss = LinFaiss()
            
    #         # 使用 WebBaseLoader 从网页加载数据
    #         loader = WebBaseLoader([web_url])
    #         data = loader.load()
    #         file_path = os.path.join('FAISS_Data', data.name)
            
    #         with open(file_path, 'wb') as file_obj:
    #             for chunk in data.chunks():
    #                 file_obj.write(chunk)
            
    #         # 根据需要，你可能需要进行数据预处理或文本拆分
    #         text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    #         docs = text_splitter.split_documents(data)
            
    #         # 获取数据的向量表示
    #         # vector_store = lin_faiss.get_embeddings([data])
            
    #         # # 保存向量和分类到数据库（你需要在 LinFaiss 类中实现相应的方法）
    #         # lin_faiss.save_to_database(vector_store)
            
    #         return Response({'success': True, 'message': 'Web URL uploaded and processed successfully'})
    #     else:
    #         return Response({'success': False, 'error': 'Invalid data'})

    @action(detail=False, methods=['post', 'get'])
    def upload_local_file(self, request):
        text_name = request.data.get('text_name')
        file = request.FILES.get('file')
        categories = request.data.get('categories')

        file_name = file
        text_path = os.path.join( 'Faiss_app', 'FAISS_Data', str(file_name))

        with open(text_path, 'wb') as text_file:
            for chunk in file.chunks():
                text_file.write(chunk)

        text_data = Faiss_data.objects.create(
            text_name=text_name,
            text=file,
            categories=categories,
        )

        # LinFaiss 实例化
        lin_faiss = LinFaiss()

        # 读取本地 txt 文件内容
        with open(text_path, 'rb') as file:
            file_content = text_file.read()

        # 根据需要，你可能需要进行数据预处理或文本拆分
        text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        docs = text_splitter.split_documents([file_content])

        # 获取数据的向量表示
        vector_store = lin_faiss.get_embeddings(docs)
        print('----------------')
        print(vector_store.length)
        print(vector_store)
        # 保存向量和分类到数据库
        lin_faiss.save_to_database(vector_store)  # 请在 LinFaiss 类中实现此方法

        serializer = self.get_serializer(text_data)

        return Response(serializer.data, {'success': True, 'message': 'Local file uploaded and processed successfully'})

