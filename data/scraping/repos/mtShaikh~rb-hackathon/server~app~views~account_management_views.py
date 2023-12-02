# Standard Library imports
import requests
# Core Flask imports
from flask import request, redirect, url_for

# Third-party imports
from pydantic import ValidationError
from flask_login import login_user, logout_user, login_required, current_user

# App imports
from ..services import account_management_services, platform_services
from ..services.platform_services import generate_user_token
from ..utils import custom_errors, sanitization
from ..utils.error_utils import (
    get_business_requirement_error_response,
    get_validation_error_response,
    get_db_error_response,
    get_authentication_failed_error
)
import random
import os
import openai



def register_account():
    unsafe_username = request.json.get("username")
    unsafe_email = request.json.get("email")
    unhashed_password = request.json.get("password")

    sanitized_username = sanitization.strip_xss(unsafe_username)
    sanitized_email = sanitization.strip_xss(unsafe_email)

    try:
        user_model = account_management_services.create_account(
            sanitized_username, sanitized_email, unhashed_password
        )
    except ValidationError as e:
        return get_validation_error_response(validation_error=e, http_status_code=422)
    except custom_errors.EmailAddressAlreadyExistsError as e:
        return get_business_requirement_error_response(
            business_logic_error=e, http_status_code=409
        )
    except custom_errors.InternalDbError as e:
        return get_db_error_response(db_error=e, http_status_code=500)

    login_user(user_model, remember=True)

    return {"message": "success"}, 201


def login_account():
    unsafe_email = request.json.get("email")
    password = request.json.get("password")

    sanitized_email = sanitization.strip_xss(unsafe_email)

    try:
        user_model = account_management_services.verify_login(sanitized_email, password)
    except ValidationError as e:
        return get_validation_error_response(validation_error=e, http_status_code=422)
    except custom_errors.CouldNotVerifyLogin as e:
        return get_business_requirement_error_response(
            business_logic_error=e, http_status_code=401
        )

    login_user(user_model, remember=True)
    token = generate_user_token(user_model)



    return {"message": "success", "token" : token}


def generate_post():
    hard_coded_data = {
      "trending_topics" : {
        "Lahore Rain": [],
        "Friday Vibes" : [
            """
            Friday Vibes

Have a blessed one y’all
            """
        ],
        "The Ashes 2023": [],
        "Facebook Threads": [
            "ELONMUSK vs MARK ZUCKERBERG MMA FIGHT :)",
            """
            Comparison of Twitter and #Threads functionality
            Twitter Owner : Elon Musk
            Threads Owner : Mark Zukerberg
            Twitter post length : 280 characters
            Threads post length : 500 characters
            """,

        ]
      }
    }

    if request.user == None:
        return get_authentication_failed_error()

    OPENAI_KEY = os.environ['OPENAI_KEY']
    openai.api_key = OPENAI_KEY

    selected_topic = random.choice(list(hard_coded_data.get("trending_topics")))
    example_posts = hard_coded_data.get("trending_topics").get(selected_topic)

    prompt_chat = "Hey chatgpt can you write me a post related to "+selected_topic

    if len(example_posts) > 0:
        prompt_chat+="""
        
        Here is examples of posts related to this topic
        
        """

        for post in example_posts:
            prompt_chat += f"""
            Example : {post}
            """

    response = openai.Completion.create(
        engine='text-davinci-003',  # Specify the engine for ChatGPT
        prompt=prompt_chat,  # The message or conversation prompt
        max_tokens=50  # Maximum length of the response
    )

    reply = response.choices[0].text.strip()

    print(prompt_chat)

    prompt_image = f"""hey for my facebook post can you generate an image to post

my post topic is {selected_topic}

and here is my post content

{reply}

"""

    response = openai.Image.create(
        prompt=prompt_image,
        n=1,
        size="256x256"
    )
    image_url = response['data'][0]['url']

    # image_data = requests.get(image_url).content
    # image = Image.open(BytesIO(image_data))

    return {
               "message": "success",
                "text": reply,
                "image_url": image_url
           }, 201


def logout_account():
    logout_user()
    return redirect(url_for("index"))


@login_required
def user():
    user_profile_dict = account_management_services.get_user_profile_from_user_model(
        current_user
    )
    return {"data": user_profile_dict}


@login_required
def email():
    unsafe_email = request.json.get("email")

    sanitized_email = sanitization.strip_xss(unsafe_email)

    try:
        account_management_services.update_email(current_user, sanitized_email)
    except ValidationError as e:
        return get_validation_error_response(validation_error=e, http_status_code=422)
    except custom_errors.EmailAddressAlreadyExistsError as e:
        return get_business_requirement_error_response(
            business_logic_error=e, http_status_code=409
        )
    except custom_errors.InternalDbError as e:
        return get_db_error_response(db_error=e, http_status_code=500)

    return {"message": "success"}, 201

# @login_required
def authorize_twitter():
    url =  platform_services.request_token_from_twitter()
    return {"url": url}, 200

# @login_required
def get_access_token():
    url = request.json.get("url")
    return platform_services.get_access_token(url)
