import requests
# import cohere
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.response import Response
from django.conf import settings
from django.contrib.auth import get_user_model


User = get_user_model()

# Remember to add login_required decorator here

# News API client library to fetch news based on the user's selected sources
class UserFinancialNewsView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [SessionAuthentication, TokenAuthentication]
    def get(self, request):
        # api_key = ''
        # co = cohere.Client(api_key)
        # counter = 0
        user = request.user
        sources = user.preference.tags
        #  https://newsdata.io/api/1/news?apikey=YOUR_API_KEY
        news_api_url = f'https://newsdata.io/api/1/news?apiKey={settings.NEWS_API_KEY}&category={",".join(sources)}&language=en'
        response = requests.get(news_api_url)
        data = response.json()
        # Test for co:here
        # for result in data["results"]:
        #     if counter < 5:
        #         content = result["content"]
        #         summary = co.summarize(content, model='summarize-xlarge', length='short')
        #         data['summary'] = summary
        #         counter += 1
        #     else:
        #         break
        return Response(data)