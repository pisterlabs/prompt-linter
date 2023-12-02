from django.urls import path

from open_ai.views import OpenAIResponseView

urlpatterns = [
    path('openai-response/', OpenAIResponseView.as_view(), name='register'),
]
