from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from utils import get_db_handle, check_password
import json

import openai
import os

default_api_key = "DEFAULT_API_KEY"
api_key = os.environ.get("AI_KEY", default_api_key)
openai.api_key = api_key

def query_chatgpt(query):
    if api_key == default_api_key:
       raise ValueError("OpenAI API key not set. Please set the AI_KEY environment variable.")
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=query,
        temperature=0,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    message = response.choices[0].text.strip()
    return message


@api_view(['GET'])
def applications(request):
    try:
        # Get the username of the logged-in user from the request parameters
        username = request.GET.get('username')

        # Query the applications collection for all applications with the specified username
        db, _ = get_db_handle("HRS")
        collection = db['applications']
        results = collection.find({"username": username})
        results = list(results) if results.count() > 0 else []

        # Return the applications data as a JSON response
        return Response({"data": results}, status=status.HTTP_200_OK)
    except Exception as e:
        print(e)  # Log the exception (optional)
        return Response({"data": []}, status=status.HTTP_200_OK)


@api_view(['POST'])
def relevantpostings(request):
    try:
        user_provided_postings = request.data.get('applications', [])

        db, _ = get_db_handle("HRS")
        collection = db['job_postings']

        # Get a random sample of 15 job postings
        random_postings = list(collection.aggregate([{'$sample': {'size': 15}}]))

        # Format the postings into a string representation for the query
        formatted_random_postings = ", ".join([f"({i+1}) {posting['title']}" for i, posting in enumerate(random_postings)])
        formatted_user_provided_postings = ", ".join([f"({i+1}) {posting['title']}" for i, posting in enumerate(user_provided_postings)])

        # Construct the query for ChatGPT
        query = f"Which of these postings is the most relevant? Return a structured response in the format of [index, index, index] where the indexes are the indexes of the list I provided you? Postings: {formatted_random_postings}. Context: {formatted_user_provided_postings}"

        # Call the query_chatgpt function with the query
        chatgpt_response = query_chatgpt(query)

        relevant_indexes = [int(i) for i in re.findall(r'\d+', chatgpt_response)]

        # Get the relevant job postings using the indexes
        relevant_postings = [random_postings[i - 1] for i in relevant_indexes]  # Subtract 1 to account for 1-based indexing in the query

        # Return the relevant job postings as a JSON response
        return Response({"data": relevant_postings}, status=status.HTTP_200_OK)
    except Exception as e:
        print(e)  # Log the exception (optional)
        return Response({"data": []}, status=status.HTTP_200_OK)
