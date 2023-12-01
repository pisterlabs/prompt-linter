"""
URL mappings for the OpenAI API.
"""
from django.urls import (
    path,
)
from openai_app import views

app_name = 'openai'

urlpatterns = [
    path('models/', views.ModelListAPIView.as_view(), name='model-list'),
    path('models/<str:model>/',
         views.ModelAPIView.as_view(), name='model-detail'),
    path('chat/completions/', views.DeductibleChatCompletionAPIView.as_view(),
         name='chat-completion'),
]
