from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
import requests
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
ESV_API_KEY = os.getenv("ESV_API_KEY")


def get_verse(request):
    return HttpResponse("HELLO")


def get_chapter(request, book, chapter):
    API_ENDPOINT = "https://api.esv.org/v3/passage/text/"
    headers = {"Authorization": "Token " + ESV_API_KEY}

    response = requests.get(
        API_ENDPOINT,
        headers=headers,
        params={
            "q": f"{book} {chapter}",
            "include-footnotes": False,
            "include-headings": False,
            "include-verse-numbers": False,
            "include-passage-references": False,
            "include-audio-link": False,
        },
    )

    return JsonResponse(response.json())


def get_book(request, book):
    API_ENDPOINT = "https://api.esv.org/v3/passage/text/"

    headers = {"Authorization": "Token " + ESV_API_KEY}
    response = requests.get(
        API_ENDPOINT,
        headers=headers,
        params={
            "q": f"{book}",
            "include-footnotes": False,
            "include-headings": False,
            "include-verse-numbers": False,
            "include-passage-references": False,
            "include-audio-link": False,
        },
    )

    return JsonResponse(response.json())


def get_book_summary(request, book):
    co = cohere.Client(COHERE_API_KEY)
    response = co.generate(
        prompt=f"Summarize the Christian book of the Bible, {book}, in 1 sentence. Include who the author is. Don't include any questions in your response.",
        max_tokens=100,
    )
    return HttpResponse(response)
