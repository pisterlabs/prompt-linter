from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .viewsets import OpenAiViewSet

router = DefaultRouter()
router.register("", OpenAiViewSet, basename="openai")


urlpatterns = [
    path("", include(router.urls)),
    
]
