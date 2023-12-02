from django.urls import path
from openai_plus.views import query

urlpatterns = [
    path('query/', query.QueryViewSet.as_view(), name='query'),
]