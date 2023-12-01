import cohere
import numpy as np
from easygoogletranslate import EasyGoogleTranslate
from rest_framework import status
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response

from commons.secret import COHERE_KEY, BASE_PROMPT
from user.models import Advisor
from user.serializers.features import AdvisorSerializer

co = cohere.Client(COHERE_KEY)
translator_id_to_en = EasyGoogleTranslate(
    source_language='id',
    target_language='en',
    timeout=10
)

translator_en_to_id = EasyGoogleTranslate(
    source_language='en',
    target_language='id',
    timeout=10
)


class ChatbotAPIView(GenericAPIView):
    def post(self, request):
        user_prompt = request.data['user_prompt']
        language = request.data['language']

        if language == 'id':
            user_prompt = translator_id_to_en.translate(user_prompt)

        prompt = BASE_PROMPT + f"The user's response is as follows: \n'{user_prompt}'"

        response = co.generate(
            prompt=prompt,
            max_tokens=64,
            temperature=0.25,
            frequency_penalty=0.6,
            presence_penalty=0.6,
        )

        response_text = response.generations

        translated_response = response_text[0]
        if language == 'id':
            translated_response = translator_en_to_id.translate(response_text[0])

        return Response({"message": translated_response}, status=status.HTTP_200_OK)


class MatchmakingAPIView(GenericAPIView):
    def calculate_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    def post(self, request):
        user_prompt = request.data.get('user_prompt')
        all_advisors = Advisor.objects.all()
        # Create a list to store each advisor's expertise
        advisors_expertise = []
        advisors_usernames = []

        for advisor in all_advisors:
            # Get all expertise associated with the advisor
            expertise = advisor.expertise.all()

            # Create a string that lists all expertise for the advisor
            expertise_list = ", ".join([str(e) for e in expertise])

            # Store the advisor's expertise and username
            advisors_expertise.append(expertise_list)
            advisors_usernames.append(advisor.user.username)


        # Create the phrases list for the co.embed call
        phrases = [user_prompt] + advisors_expertise

        # Get embeddings for all phrases in a single API call
        embeddings = co.embed(phrases).embeddings
        user_embed = embeddings[0]  # user's embedding
        advisors_embeds = embeddings[1:]  # advisors' embeddings

        # Calculate similarity scores for each advisor
        advisors_scores = []
        for username, advisor_embed in zip(advisors_usernames, advisors_embeds):
            # Calculate similarity score
            score = self.calculate_similarity(user_embed, advisor_embed)

            # Store the advisor and their score in the list
            advisors_scores.append((username, score))

        # Sort the advisors by their similarity score in descending order
        advisors_scores.sort(key=lambda x: x[1], reverse=True)

        # Get the top 3 advisors
        top_advisors = advisors_scores[:3]

        # Get the Advisor objects for the top 3 advisors
        top_advisors_objects = Advisor.objects.filter(user__username__in=[advisor[0] for advisor in top_advisors])

        # Serialize the top 3 advisors
        top_advisors_serializer = AdvisorSerializer(top_advisors_objects, many=True)

        # Return the serialized data
        return Response({'top_advisors': top_advisors_serializer.data})
