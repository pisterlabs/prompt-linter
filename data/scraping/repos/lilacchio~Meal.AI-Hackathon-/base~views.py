from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.http import HttpResponse
from .models import Meal

import os
import openai
openai.api_key = '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'

# Create your views here.

def home(request):
    return render(request, 'base/home.html')

def loginPage(request):
    page = 'loginP'
    context = {'page': page}
    if (request.user.is_authenticated):
        return redirect('home')
    if (request.method=="POST"):
        username = request.POST.get('username').lower()
        password = request.POST.get('password')
        try:
            user = User.objects.get(username=username)
        except:
            messages.error(request, 'User does not exist')
        user = authenticate(request, username=username, password=password)
        if (user is not None):
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Incorrect username or password')
                    
    return render(request, 'base/login_register.html', context)

def logoutUser(request):
    logout(request)
    return redirect('home')

def registerUser(request):
    form = UserCreationForm()
    if (request.method=='POST'):
        form = UserCreationForm(request.POST)
        if (form.is_valid):
            user = form.save(commit=False)
            user.username = user.username.lower()
            user.save()
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'An error occured, please try again')
    context = {'form': form}
    return render(request, 'base/login_register.html', context)

def newMeal(request):
    if (request.method=="POST"):
        bmi = request.POST.get('bmi')
        location = request.POST.get('location')
        budget = request.POST.get('budget')
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt="Generate a healthy and sustainable meal plan for {} BMI in {} within {}".format(bmi, location, budget),
            temperature=0.7,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        content = response['choices'][0]['text']
        meal = Meal(user=request.user, content=content, bmi=int(bmi), location=location, budget=int(budget))
        meal.save()
    return render(request, 'base/newMeal.html', {'meal': meal})

def userMeals(request):
    meals = Meal.objects.filter(user=request.user)
    context = {'meals': meals}
    print (meals)
    return render(request, 'base/userMeals.html', context)

def impact(request):
    return render(request, 'base/impact.html')

def about(request):
    return render(request, 'base/about.html')