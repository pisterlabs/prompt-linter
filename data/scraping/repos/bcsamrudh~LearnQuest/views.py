from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render,redirect
from .models import Notes
from .forms import NotesForm
from django.contrib import messages
from django.urls import reverse
from django.db.models import Q
from django.conf import settings
from django.contrib.auth.decorators import login_required   
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.contrib.auth import get_user_model
from django.views.decorators.cache import never_cache
import datetime
import google.generativeai as palm
import cohere

User_model = get_user_model()

@never_cache
def about(request):
    return render(request,"about.html")

@login_required
def generate_questions(request,topic,subject):
    palm.configure(api_key=settings.API_KEY)

    defaults = {
    'model': 'models/text-bison-001',
    'temperature': 0.7,
    'candidate_count': 1,
    'top_k': 40,
    'top_p': 0.95,
    'max_output_tokens': 1024,
    'stop_sequences': [],
    'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":1},{"category":"HARM_CATEGORY_TOXICITY","threshold":1},{"category":"HARM_CATEGORY_VIOLENCE","threshold":2},{"category":"HARM_CATEGORY_SEXUAL","threshold":2},{"category":"HARM_CATEGORY_MEDICAL","threshold":2},{"category":"HARM_CATEGORY_DANGEROUS","threshold":2}],
    }
    prompt = prompt = f"""Please generate 10 related follow-up questions of medium difficulty and answers to this topic: {topic} of subject {subject}"""

    response = palm.generate_text(
    **defaults,
    prompt=prompt
    )
    data = response.result
    context={"data":data,"topic":topic,"subject":subject}
    return render(request,'notes/questions_view.html',context)


@login_required
def upload_notes(request):
    if request.method=='POST':
        form=NotesForm(request.POST,request.FILES)
        if form.is_valid():
            subject = form.cleaned_data.get('subject')
            title = form.cleaned_data.get('title')
            notesfile = form.cleaned_data.get('notesfile')
            university = form.cleaned_data.get('university')
            filetype = notesfile.name.split('.')[1].upper()
            description = form.cleaned_data.get('description')
            tags = form.cleaned_data.get('tags')
            user =User_model.objects.filter(username=request.user.username).first()
            user.consistency_score+=50
            user.save()
        try:
            Notes.objects.create(title=title,user=user,subject=subject,notesfile=notesfile,filetype=filetype,description=description,tags=tags,university=university)
            return redirect(reverse('notes',kwargs={'search':0}))
        except:
            messages.error(request,"Error occured while uploading Notes, Please try again!")
    else:
        form = NotesForm(None)
    return render(request,'notes/upload_notes.html',{"form":form})

def my_notes(id):
    user = get_object_or_404(User_model,id=id)
    notes = Notes.objects.filter(user=user)
    return notes

    
# @login_required
# def update_notes(request, slug):
#     obj = get_object_or_404(Notes, slug=slug)
#     print(request.FILES)
#     form = NotesForm(request.POST or None,request.FILES or None,instance = obj)
#     if form.is_valid():
#         form.save()
#         return redirect(reverse('note',kwargs={'slug':slug}))
#     return render(request,'notes/upload_notes.html',{"form":form})

@login_required
def delete_notes(request,slug):
    notes = get_object_or_404(Notes,slug=slug)
    notes.delete()
    return redirect(reverse('notes',kwargs={"search":0}))


def is_valid_queryparam(param):
    return param != '' and param is not None
    
def filter(request):
    qs = Notes.objects.all()
    title_contains_query = request.POST.get('title_contains')
    user_query = request.POST.get('user')
    date_query = request.POST.get('date_query')
    subject = request.POST.get('subject')
    tags_query = request.POST.get('tags')

    if is_valid_queryparam(title_contains_query):
        qs = qs.filter(title__icontains=title_contains_query)

    if is_valid_queryparam(user_query):
        qs = qs.filter(Q(user__username__icontains=user_query)).distinct()

    if is_valid_queryparam(date_query):
        date = datetime.datetime.strptime(date_query, '%Y-%m-%d').date()
        qs = qs.filter(date_uploaded__year=date.year,date_uploaded__month=date.month,date_uploaded__day=date.day)

    if is_valid_queryparam(subject):
        qs = qs.filter(subject__icontains=subject)

    if is_valid_queryparam(tags_query):
        qs = qs.filter(tags__icontains=tags_query)

    return qs

@login_required
def notes(request,search):
    if search:
        notes=filter(request)
    else:
        notes=Notes.objects.all()
    p = Paginator(notes.order_by('-date_uploaded'), 5)  
    page_number = request.POST.get('page')
    try:
        page_obj = p.get_page(page_number)

    except PageNotAnInteger:
        page_obj = p.page(1)

    except EmptyPage:
         page_obj = p.get_page(p.num_pages)
         
    context = {'page_obj': page_obj}
    return render(request, 'notes/view_notes.html', context)


@login_required
def note(request,slug):
    try:
        notes_display=get_object_or_404(Notes,slug=slug)
        if notes_display.filetype == "PDF":
            filetype = "PDF"
        else:
            filetype="None"
    except Notes.DoesNotExist:
        return render(request, '404.html')
    except :
        return render(request,'404.html')
    context={"note": notes_display,"current_user":request.user,"filetype":filetype}
    return render(request,'notes/note.html',context=context)


def upvote(request,id):
        note_obj = get_object_or_404(Notes, id=id)
        if note_obj.upvotes.filter(id=request.user.id).exists(): 
            note_obj.upvotes.remove(request.user) 
            note_obj.save()
            text = True
        else:
            note_obj.upvotes.add(request.user) 
            note_obj.save()
            text = False
        return JsonResponse({'total_upvotes': note_obj.total_upvotes,'text':text})
 