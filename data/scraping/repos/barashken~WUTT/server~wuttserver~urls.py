"""
URL configuration for wuttserver project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from search.views import OpenAIView
from users.views import RegisterView, LoginView, LogoutView
from trips.views import GetTripView
from .views import check_alive

urlpatterns = [
    path('admin/', admin.site.urls),
    path('check-alive/', check_alive, name='check-alive'),
    path('call-openai-api/', OpenAIView.as_view(), name='call-openai-api'),
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('get-trip/<int:id>/', GetTripView.as_view(), name='get-trip'),
]
