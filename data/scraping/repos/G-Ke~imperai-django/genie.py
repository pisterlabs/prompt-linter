from rest_framework import viewsets, permissions, status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from drf_spectacular.utils import extend_schema, OpenApiExample, OpenApiParameter, OpenApiTypes
from rest_framework import serializers
from rest_framework.views import APIView
from .vertex import global_openapi_parameters
import cohere
import os

co = cohere.Client(os.environ.get('COHERE_API_KEY'))


class ChatRequestSerializer(serializers.Serializer):
    message = serializers.CharField(max_length=1000)

"""parameters=[
    OpenApiParameter(name='Message', description='A Message to Send', required=True, type=str),
]"""

@extend_schema(
        request=ChatRequestSerializer,
        responses={200: ChatRequestSerializer},
        tags=['Genie v1'], 
        parameters=global_openapi_parameters,
        examples=[
            OpenApiExample(
                'Message Only Request',
                value={'message': 'What sort of tasks can you help with?'}
            )
        ]
    ) 
@api_view(['post'])
def make_cohere_chat_beta_request(request):
    serializer = ChatRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    message = serializer.validated_data['message']
    response = co.chat(
        message=message,
    )
    if response:
        return Response(response.text, status=status.HTTP_200_OK)
    else:
        return Response('Error', status=status.HTTP_400_BAD_REQUEST)