import os
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import JSONParser, MultiPartParser
from .services.OpenAIServices import OpenAIService
from .serializers import CreateChatCompletionSerializer, CreateCompletionSerializer, CreateEditsSerializer, \
    CreateImageSerializer, CreateTranscriptionSerializer, CreateTranslationSerializer


class OpenAiViewSet(viewsets.GenericViewSet):
    parser_classes = (JSONParser, MultiPartParser)
    openai_service = OpenAIService(
        base_url=os.getenv('OPENAI_BASE_URL', "https://api.openai.com"),
        openai_api_key=os.getenv('OPENAI_API_KEY', ""),
    )

    allowed_serializer = {
        "create_a_chat_completion": CreateChatCompletionSerializer,
        "create_a_completion": CreateCompletionSerializer,
        "create_an_edit": CreateEditsSerializer,
        "create_an_image_generations": CreateImageSerializer,
        "create_an_audio_translation": CreateTranscriptionSerializer,
        "create_an_audio_transcription": CreateTranslationSerializer
    }

    def get_serializer_class(self):
        return self.allowed_serializer.get(self.action)

    @action(detail=False, methods=['get'], url_path='models')
    def get_models_list(self, request):
        """
        To get the models list
        :return: Returns all list of models.
        """
        response = self.openai_service.list_models()
        return Response(data=response.get("data"), status=response.get("status_code"))

    @action(detail=False, methods=['get'], url_path='models/(?P<model_id>[^/.]+)')
    def get_model_detail(self, request, model_id):
        """
        To get the models details
        :path_params: model_id
        :return: Returns models details.
        """
        response = self.openai_service.retrieve_model(model_id)
        return Response(data=response.get("data"), status=response.get("status_code"))

    @action(detail=False, methods=['get'], url_path='engines')
    def get_engine_list(self, request):
        """
        To get the engine list
        :return: Returns list of engine.
        """
        response = self.openai_service.list_engine()
        return Response(data=response.get("data"), status=response.get("status_code"))

    @action(detail=False, methods=['get'], url_path='engines/(?P<engine_id>[^/.]+)')
    def get_engine_detail(self, request, engine_id):
        """
        To get the engine details'
        :path_params: engine_id
        :return: Returns models details.
        """
        response = self.openai_service.retrieve_engine(engine_id)
        return Response(data=response.get("data"), status=response.get("status_code"))

    @action(detail=False, methods=['post'], url_path='chat/completions')
    def create_a_chat_completion(self, request):
        """
        To create a chat completion
        :payload: {"model": "", "messages": ["role": "", "content": ""]}
        :return: Returns a newly created chat completion detail.
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        response = self.openai_service.create_chat_completion(payload=serializer.data)
        return Response(data=response.get("data"), status=response.get("status_code"))

    @action(detail=False, methods=['post'], url_path='completion')
    def create_a_completion(self, request):
        """
        To create a completion
        :payload: {"model": "", "prompt": "", "max_tokens": "", "temperature": ""}
        :return: Returns a newly created completion.
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        response = self.openai_service.create_completion(payload=serializer.data)
        return Response(data=response.get("data"), status=response.get("status_code"))

    @action(detail=False, methods=['post'], url_path='edits')
    def create_an_edit(self, request):
        """
        To create an edit
        :payload: {"model": "", "input": "", "instruction": ""}
        :return: Returns a newly created edit with detail.
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        response = self.openai_service.create_edit(payload=serializer.data)
        return Response(data=response.get("data"), status=response.get("status_code"))

    @action(detail=False, methods=['post'], url_path='images/generations')
    def create_an_image_generations(self, request):
        """
        To create an edit
        :payload: {"prompt": "", "n": int, "size": ""}
        :return: Returns a newly created image url.
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        response = self.openai_service.create_image(payload=serializer.data)
        return Response(data=response.get("data"), status=response.get("status_code"))

    @action(detail=False, methods=['post'], url_path='audio/transcription')
    def create_an_audio_transcription(self, request):
        """
        To create a transcription
        :payload: {"file": "", "model": ""}
        :return: Returns a newly created transcription text.
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        response = self.openai_service.create_transcription(payload=serializer.validated_data)
        return Response(data=response.get("data"), status=response.get("status_code"))

    @action(detail=False, methods=['post'], url_path='audio/translation')
    def create_an_audio_translation(self, request):
        """
        To create a translation
        :payload: {"file": "", "model": ""}
        :return: Returns a newly created translation text.
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        response = self.openai_service.create_translation(payload=serializer.validated_data)
        return Response(data=response.get("data"), status=response.get("status_code"))