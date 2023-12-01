"""
URL configuration for fmlOps project.

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
from chromepipeline import views as chromeviews
from visualization import views as visviews
from redditInfo import views as redditviews
from langchainbot import views as langchainviews

urlpatterns = [
    path("admin/", admin.site.urls),
    path('downloads/', chromeviews.downloads),
    path('routine/', chromeviews.routines),
    path('visual/', visviews.chart_view, name='visualization'),
    path('reddit/', redditviews.redditProcessing, name='reddit_data'),
    path('mysearch/', langchainviews.searchQuery, name='search_query'),
    path('langchain/', langchainviews.redditQuery),
    path('display/', langchainviews.displayRedditData)
]
