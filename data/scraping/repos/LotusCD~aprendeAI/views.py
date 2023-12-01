# views.py
from django.shortcuts import render
from django.conf import settings
from openai import OpenAI

# Initialize OpenAI client using settings
client_api = OpenAI(api_key=settings.OPENAI_API_KEY)

def query_openai_api(prompt):
    try:
        response = client_api.completions.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text
    except Exception as e:
        print("Error querying OpenAI API: ", e)
        return "Error occurred while processing your request."

def index(request):
    if request.method == 'POST':
        user_input = request.POST.get('userInput')
        api_response = query_openai_api(user_input)
        return render(request, 'index.html', {'userInput': user_input, 'apiResponse': api_response})
    else:
        return render(request, 'index.html')
