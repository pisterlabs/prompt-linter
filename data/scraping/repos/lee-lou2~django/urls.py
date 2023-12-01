from django.urls import path, include
from rest_framework.routers import DefaultRouter

from app.openai.v1.views import OpenaiChatViewSet

router = DefaultRouter()
router.register("chat", OpenaiChatViewSet)

urlpatterns = [
    path("openai/", include(router.urls)),
]
