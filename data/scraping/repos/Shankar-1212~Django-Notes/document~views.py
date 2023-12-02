from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Document
import openai

api_key = ""
openai.api_key = api_key

def editor(request):
    docid = int(request.GET.get('docid', 0))
    documents = Document.objects.all()

    if request.method == 'POST':
        docid = int(request.POST.get('docid', 0))
        title = request.POST.get('title')
        content = request.POST.get('content', '')

        if docid > 0:
            document = Document.objects.get(pk=docid)
            document.title = title
            document.content = content
            document.save()

            return redirect('/?docid=%i' % docid)
        else:
            document = Document.objects.create(title=title, content=content)

            return redirect('/?docid=%i' % document.id)
    if docid > 0:
        document = Document.objects.get(pk=docid)
    else:
        document = ''

    context = {
        'docid': docid,
        'documents': documents,
        'document': document
    }

    return render(request, 'editor.html', context)

def delete_document(request, docid):
    document = Document.objects.get(pk=docid)
    document.delete()

    return redirect('/?docid=0')

def generate_text(request):
    if request.method == 'POST' and 'generate' in request.POST:
        # Retrieve the document title (similar to the existing logic in your editor view)
        docid = int(request.POST.get('docid', 0))
        document = Document.objects.get(pk=docid) if docid > 0 else None
        document_title = document.title if document else ""

        # Use the document title as input for text generation
        chat_message = {
            "role": "user",
            "content": document_title
        }

        # Generate text using OpenAI ChatCompletion API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            messages=[chat_message],
        )

        # Print the response for debugging
        print(response)  # This will show the API response in your console

        # Check for any errors in the response
        if 'error' in response:
            return HttpResponse("Error: " + response['error']['message'])

        # Extract the generated text from the API response
        generated_text = response['choices'][0]['message']['content']

        # Insert the generated text into the content field of the document
        if document:
            document.content = generated_text
            document.save()

        # Redirect back to the editor page
        return redirect('/?docid=%i' % docid)

    # Handle other cases, such as GET requests (if needed)
    return HttpResponse("Invalid request")
