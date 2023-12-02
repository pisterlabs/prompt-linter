from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings

import os
import openai
openai.organization = "org-mYhYA5YU0A9wqejVqBtNNVkY"
openai.api_key = os.getenv("OPENAI_API_KEY")


class WordAPIView(APIView):
    def get(self, request):
        return Response({'message': 'hello words'})



class WordFromTextAPIView(APIView):
    language_list = settings.LANGUAGE_LIST
    
    def post(self, request):
        user = request.user
        # if user.is_anonymous:
        #     return Response({'message': 'You must be logged in to use this feature.'}, status=401)

        text = request.data['text']
        target_language = self.language_list[user.profile.target_language]
        primary_language = self.language_list[user.profile.primary_language]
        card_format = user.profile.card_front + ': ' + user.profile.card_back

        prompt = 'You will now create word flashcards for language learning.' \
            'Read a sentence in {}, the language you are learning, and list the words and phrases and their meanings in your native language, {}.'  \
            'Choose words and phrases that are difficult or not usually used very often. The format is absolute and must be followed, do not add anything.' \
            'Answers other than the word list will not be accepted. [] represents variables. Insert the variable you think it is. ' \
            'format: {} ' \
            'EXAMPLE: [documents: 書類 (Please submit all the required documents.)]' \
            .format(target_language, primary_language, card_format)
        

        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo-0613",
            temperature = 0.9,
            max_tokens = 200,
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )

        words = response.choices[0].message.content.split('\n')

        cards = []
        for word in words:
            if len(word) > 0:
                cards.append({"front": word.split(':')[0], "back": word.split(':')[1]})

        return Response({'cards': cards})