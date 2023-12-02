from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.utils.safestring import mark_safe
from django.http import JsonResponse
from .models import Blog
import json
from Blog_Generator.functions import (
    email_check,
    get_youtube_audio,
    get_youtube_title,
    get_youtube_transcription,
    generate_blog_from_cohere,
    is_password_up_to_standard,
    content_formatter)


# Create your views here.
@login_required
def index(request):
    """ This function returns or render the index
        page of this Application
    """
    return render(request, 'index.html')


def user_login(request):
    """ The function that handles all users login
        operation
    """
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('/')
        else:
            return render(request, 'login.html',
                          {'error_message': 'Invalid cridentials'})

    return render(request, 'login.html')


def user_signup(request):
    """ This function will handle user signup
    """
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        repeat_password = request.POST['repeatpassword']

        # If email is valid
        if not email_check(email):
            return render(request, 'signup.html',
                          {'error_message': 'Empty or invalid email'})

        # Check if user already exists
        if User.objects.filter(username=username).exists():
            return render(request, 'signup.html',
                          {'error_message': 'User already exists'})

        # Check if email already exists
        if User.objects.filter(email=email).exists():
            return render(request, 'signup.html',
                          {'error_message': 'Email already exists'})

        # Check if password is up to standard
        if not is_password_up_to_standard(password):
            message = 'Please use a stronger password.\nUse at \
                    least one capital character, numbers and symbols'
            return render(request, 'signup.html',
                          {'error_message': message})

        # Check if password is and repeated password are alike
        if password != repeat_password:
            return render(request, 'signup.html',
                          {'error_message': 'Please check your password'})
        else:
            # Create user
            try:
                user = User.objects.create_user(username, email, password)
                user.save()
                login(request, user)  # User login after signup is auto
                return redirect('/')
            except Exception:
                return render(request, 'signup.html',
                              {'error_message': 'Error creating your account'})

    return render(request, 'signup.html')


def user_logout(request):
    """ This function will log user out of their when
        they click on the logout button.
    """
    logout(request)
    return redirect('/')


def session_logout_user(request):
    """ The function will automatically logout user
        after 15 minutes of inactivity
    """
    request.session.flush()
    return redirect('/')


@login_required
def saved_blog(request):
    """ This function will render the for
        saved blog post done by the user.
    """
    user_blog = Blog.objects.filter(user=request.user)
    if not user_blog:
        return render(request, 'saved_blog.html',
                      {'error_message': 'No Blog Found'})

    return render(request, 'saved_blog.html',
                  {'user_blog': user_blog})


@login_required
def blog_posts(request, pg):
    """ This function will use each blog post index or id
        to render the respective blog post that he user
        request for by cliking on the list of previous saved blogs
    """
    user_blog = Blog.objects.get(id=pg)
    if request.user == user_blog.user:
        return render(request, 'blog_content.html',
                      {'user_blog': user_blog})
    else:
        return redirect('/')


@csrf_exempt
def blog_content(request):
    """ This function contains the all logic require to
        process the user link input, this function will
        make calls to all necessary endpoint or other functions
        and make sure that the program execute properly even when
        error occurred
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            youtubelink = data['link']
            title = get_youtube_title(youtubelink)

            # Get Transcript
            transcript = get_youtube_transcription(youtubelink)
            if not transcript:
                message = 'Transcription could not be retrieved'
                return JsonResponse({'error': message}, status=400)

            # Get Generated Blog Contents From CohereAI
            gen_content = generate_blog_from_cohere(transcript)
            blog_content = mark_safe(gen_content)
            if not blog_content:
                message = 'Unable to generate blog content'
                return JsonResponse({'error': message}, status=400)

            if len(blog_content) < 40:
                message = 'I am sorry, I could not generate a blog \
                        post for you from the link you provided. \
                        Please try again'
                return JsonResponse({'content': message}, status=400)

            else:
                # If everything goes well save Blog Post
                blog_content = content_formatter(blog_content)
                user_generated_content = Blog(
                    user=request.user,
                    youtube_title=title,
                    youtube_link=youtubelink,
                    content=blog_content
                )
                user_generated_content.save()

            return JsonResponse({'content': blog_content}, status=200)
            # 'title': title, 'link': youtubelink}, status=200)

        except (json.JSONDecodeError, KeyError):
            message = 'Link not found or data could not be retireved'
            return JsonResponse({'error': message}, status=400)

    else:
        return JsonResponse({'error': 'Invalid Request'}, status=405)
