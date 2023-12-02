from django.shortcuts import render
from . import forms
import os
from datetime import datetime
import openai
from dotenv import load_dotenv
from .utils import call_api

load_dotenv()
SECRET_KEY = os.getenv('SECRET_KEY')
openai.api_key = SECRET_KEY

messages = []
user_prompt = ""

def chatbox(request):
    form = forms.InputForm(request.POST or None)
    user_msg = None     
    logged_in_user = request.user
    
    # my_form = form.save(commit=False)
    # my_form.user= request.user
    # my_form.save()
    
    if form.is_valid() :
        user_msg = form.cleaned_data['text_msg']     
        messages.append({user_msg : ""})
        form = forms.InputForm()
        time = datetime.today().strftime('%H:%M')
        
        response = call_api(user_msg)
        
        normalized_user_message = user_msg.replace(" ", "").replace(",", "").replace(".", "").lower()
        normalized_bot_message = response.replace(" ", "").replace(",", "").replace(".", "").lower()[2:]
        form = forms.InputForm()
        
        if normalized_user_message == normalized_bot_message:
            global user_prompt
            user_prompt += f"You: {user_msg}\nFriend:"
            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=user_prompt,
                temperature=0.5,
                max_tokens=600,
                top_p=1.0,
                frequency_penalty=0.5,
                presence_penalty=0.0,
                stop=["You:"]
                )
            messages.append({'': response['choices'][0]['text']})
            user_prompt += response['choices'][0]['text']
            return render(request, "chatbox/chatbox.html", {'form': form, 'messages':messages, 'time': time})

        else:
            messages.append({"" : "Your sentence is incorrect. It should be as follow:"})
            messages.append({"" : response})
            messages.append({"" : "Try to write it again to improve your skill."})
            return render(request, "chatbox/chatbox.html", {'form': form, 'messages':messages, 'time': time})

    else:
        if str(logged_in_user) != "AnonymousUser":
            messages.clear()
            messages.append({"": f"Hello {logged_in_user}, how are you doing today ?"})
        else:
            messages.clear()
        return render(request, "chatbox/chatbox.html", {'form': form, 'messages':messages, 'user_name': logged_in_user})

