import markdown
import openai
import ratelimit
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, get_object_or_404, redirect
from django.utils.decorators import method_decorator
from django.shortcuts import redirect

from accounts.forms import PermissionForm, AddPositionForm
from accounts.models import Worker, Role
from ai import process
from .forms import newChatForm
from .models import Chat, User, ChatMessage
import os


@login_required
def profile(request):
    chats = Chat.objects.filter(user=request.user)
    roles = Worker.objects.filter(user=request.user)

    if request.method == 'POST':
        form = PermissionForm(request.POST)
        form2 = AddPositionForm(request.POST)
        if form.is_valid():
            users = form.cleaned_data['users']
            permission = form.cleaned_data['permission']
            for user in users:
                user.is_superuser = permission
                user.is_staff = permission
                user.save()
            return redirect('profile')
        if form2.is_valid():
            position = form2.cleaned_data['position']
            Role.objects.create(name=position)
        return redirect('profile')
    else:
        form = PermissionForm()
        form2 = AddPositionForm()
    context = {
        'roles': roles,
        'chats': chats,
        'form': form,
        'form2': form2
    }
    return render(request, 'profile.html', context)


@login_required
def chat(request, pk):
    tchat = get_object_or_404(Chat, pk=pk)
    if request.method == 'POST':
        messages = ChatMessage.objects.filter(chat=tchat)
        ct = ""
        for message in messages:
            ct += message.message + "\n" + message.answer + "\n"
        response = process(request.POST['chat_input'], tchat.context + ct)
        konvRes = markdown.markdown(response)
        ChatMessage.objects.create(
            chat=tchat,
            message=str(request.POST.get('chat_input', False)),
            answer=konvRes,
            user=request.user
        )
        tchat.save()
        return redirect('chat', tchat.pk)
    else:
        user = request.user
        if ChatMessage.objects.filter(chat=tchat).exists():
            messages = ChatMessage.objects.filter(chat=tchat)
            return render(request, 'chat.html', {'messages': messages})
        else:
            message = {}
            return render(request, 'chat.html', message)


@login_required
def new_chat(request):
    user = get_object_or_404(User, pk=request.user.pk)
    if request.method == 'POST':
        form = newChatForm(request.POST)
        form.user = user
        if form.is_valid():
            user = form.save(commit=False)
            user.user = request.user
            user.save()
            return redirect('chat', pk=user.pk)
    else:
        form = newChatForm()
    return render(request, 'new_chat.html', {'form': form})


def uc(request):
    return render(request, 'underconstr.html')
