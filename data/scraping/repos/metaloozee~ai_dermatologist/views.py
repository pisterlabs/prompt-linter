from .forms import QuestionForm 

from django.shortcuts import render
from decouple import config
import openai

def home(request):
    if request.method == "POST":
        form = QuestionForm(request.POST)
        if form.is_valid():
            question = form.cleaned_data['question']

            openai.api_key = config('OPENAI_API_KEY')
            openai.Model.list()

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "Assuming you are an expert in dermatology, Kindly answer the following prompts by suggesting the best possible products/ingredients for the specific skin type or problem. (Try to provide me with output as a rich text format)"
                        },
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                )

                response = response.choices[0].message.content

                return render(request, 'home.html', {"form": form, "response": response})

            except Exception as e:
                return render(request, 'home.html', {"form": form, "response": e})
        else:
            return render(request, 'home.html', {"form": form})

    else:
        form = QuestionForm()
        return render(request, 'home.html', {"form": form})
