from django.shortcuts import render
from django.http import HttpResponse
'''
# Create your views here.
#This is for handling home page logic.
def home(request):
    return render(request, 'assistant/home.html')

    #return HttpResponse("This is a Home Page")


def error_handler(request):

    return render(request, 'assistant/404.html')
    #return HttpResponse("404 Error Page")
'''


# importing render and redirect
from django.shortcuts import render, redirect
# importing the openai API
import openai
# import the generated API key from the secret_key file
from .secret_key import API_KEY
# loading the API key from the secret_key file
openai.api_key = API_KEY

# this is the home view for handling home page logic
def home(request):
    # the try statement is for sending request to the API and getting back the response
    # formatting it and rendering it in the template
    try:
        # checking if the request method is POST
        if request.method == 'POST':
            # getting prompt data from the form
            prompt = request.POST.get('prompt')
            # making a request to the API
            response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=1, max_tokens=1000)
            # formatting the response input
            formatted_response = response['choices'][0]['text']
            # bundling everything in the context
            context = {
                'formatted_response': formatted_response,
                'prompt': prompt
            }
            # this will render the results in the home.html template
            return render(request, 'assistant/home.html', context)
        # this runs if the request method is GET
        else:
            # this will render when there is no request POST or after every POST request
            return render(request, 'assistant/home.html')
    # the except statement will capture any error
    except:
        # this will redirect to the 404 page after any error is caught
        return redirect('error_handler')

# this is the view for handling errors
def error_handler(request):
    return render(request, 'assistant/404.html')



from .forms import ContactForm,StudentForm
'''def contact(request):
    form= ContactForm()
    return render(request, 'assistant/form.html', {'form': form})'''

def contact(request):
    if request.method == 'POST':
        form=ContactForm(request.POST)
        if form.is_valid():
            name=form.cleaned_data['name']
            email=form.cleaned_data['email']
            category = form.cleaned_data['category']
            subject = form.cleaned_data['subject']
            body = form.cleaned_data['body']
            print(name, email, category, subject, body)
    form=ContactForm()
    return render(request,'assistant/form.html',{'form':form})

def student(request):
    form= StudentForm()
    return render(request, 'assistant/student.html', {'form': form})