"""
This module interfaces with user data.
"""

# import data.food
import os
import random
from string import ascii_uppercase
import data.db_connect as con
from PIL import Image
import pytesseract
import datetime
import openai
from google.oauth2 import id_token
from google.auth.transport import requests

TEST_USERNAME_LENGTH = 6
TEST_NAME_LENGTH = 6

NAME = 'Name'
PANTRY = 'Pantry'
USERNAME = "Username"
SAVED_RECIPES = 'Saved_Recipes'
INSTACART_USR = 'Instacart_User_Info'
GROCERY_LIST = 'Grocery List'
ALLERGENS = 'Allergens'
AUTH_EXPIRES = "Auth_Exp"


class AuthTokenExpired(Exception):
    pass


def _get_test_username():
    con.connect_db()
    username = ''.join(random.choices(ascii_uppercase, k=TEST_USERNAME_LENGTH))
    while user_exists(username):
        username = ''.join(random.choices(
            ascii_uppercase, k=TEST_USERNAME_LENGTH))
    return username


def _get_test_name():
    con.connect_db()
    name = ''.join(random.choices(ascii_uppercase, k=TEST_NAME_LENGTH))

    return name


def _get_test_exp():
    return datetime.datetime.now() + datetime.timedelta(hours=1)


def _get_test_auth_token(username="TESTING"):
    return {
        'email': username,
        'name': username,
        'exp': int((datetime.datetime.now() +
                    datetime.timedelta(hours=1)).timestamp())
    }


def user_exists(username):
    con.connect_db()
    try:
        con.fetch_one(con.USERS_COLLECTION, {USERNAME: username})
        res = True
    except ValueError:
        res = False
    return res


def _create_test_user():
    username = _get_test_username()
    name = _get_test_name()
    exp = _get_test_exp()
    print(username, name, exp)
    test_user = create_user(username, name, exp)
    return test_user


def get_users():
    con.connect_db()
    users = con.fetch_all(con.USERS_COLLECTION)
    for user in users:
        user[con.MONGO_ID] = str(user[con.MONGO_ID])
    return users


def get_user(username: str) -> str:
    con.connect_db()
    if auth_expired(username):
        raise AuthTokenExpired("User's Authentication Token is expired")
    try:
        res = con.fetch_one(con.USERS_COLLECTION, {USERNAME: username})
        res[con.MONGO_ID] = str(res[con.MONGO_ID])
    except ValueError:
        raise ValueError(f'User {username} does not exist')

    return res


def auth_expired(username: str) -> bool:
    exp = con.fetch_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {AUTH_EXPIRES: 1, con.MONGO_ID: 0}
    )

    return exp[AUTH_EXPIRES] <= datetime.datetime.now().timestamp()


def valid_authentication(google_id_token):
    # Add check for CLIENT ID for app that accesses authentication
    # Maybe save valid CLIENT ID to check against in os.environ()
    idinfo = id_token.verify_oauth2_token(google_id_token, requests.Request())

    # aud = idinfo['aud']
    # if os.environ.get("GOOGLE_CLIENT_ID") != aud:
    #     raise ValueError("Invalid Token")

    exp = idinfo['exp']
    if exp < datetime.datetime.now().timestamp():
        raise AuthTokenExpired("Expired token")
    return idinfo


def auth_user(google_id_token):
    try:
        id_info = valid_authentication(google_id_token)
        username = id_info['email']
        if not user_exists(username):
            raise ValueError("User associated with token does not exist")

        exp = datetime.datetime.fromtimestamp(id_info['exp'])

        con.update_one(
            con.USERS_COLLECTION,
            {USERNAME: username},
            {AUTH_EXPIRES: exp}
        )

    except ValueError as ex:
        # Invalid token
        raise ex
    except AuthTokenExpired as ex:
        raise ex


def generate_google_user(google_id_token):
    id_info = valid_authentication(google_id_token)

    username = id_info['email']
    name = id_info['name']
    exp = datetime.datetime.fromtimestamp(id_info['exp'])
    print(username, name, exp)
    create_user(username, name, exp)


def create_user(username: str, name: str, expires: datetime.datetime) -> dict:
    con.connect_db()
    if len(username) < 5:
        raise ValueError(f'Username {username} is too short')

    if user_exists(username):
        raise ValueError(f'User {username} already exists')

    print(type(username))
    print(type(name))

    new_user = {
        USERNAME: username,
        NAME: name,
        PANTRY: [],
        SAVED_RECIPES: {},
        INSTACART_USR: None,
        GROCERY_LIST: [],
        ALLERGENS: [],
        AUTH_EXPIRES: int(expires.timestamp()),
    }
    print(f'{new_user=}')

    add_ret = con.insert_one(con.USERS_COLLECTION, new_user)
    print(f'{add_ret}')
    return new_user


def remove_user(username):
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    del_res = con.del_one(con.USERS_COLLECTION, {USERNAME: username})

    print(f'{del_res}')
    return f'Successfully deleted {username}'


def logout_user(username):
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    con.update_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {AUTH_EXPIRES: 0}
    )


def get_pantry(username):
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    pantry_res = con.fetch_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {PANTRY: 1, con.MONGO_ID: 0}
    )

    return pantry_res


def add_to_pantry(username: str, food: list[str]) -> str:
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    con.update_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {"$push": {PANTRY: {"$each": food}}}
    )
    return f'Successfully added {food}'


def get_recipes(username):
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    recipes_res = con.fetch_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {SAVED_RECIPES: 1, con.MONGO_ID: 0}
    )

    return recipes_res


def generate_recipe(username, query):
    app_key = '274c6a9381c49bc303a30cebb49c84d4'
    app_id = '29bf3511'
    query_string = 'https://api.edamam.com/api/recipes\
        /v2?type=public&q=' + query + '&app_id=' + app_id +\
        '&app_key=' + app_key
    x = requests.get(query_string)
    return x  # return full recipe response body


def add_to_recipes(username, recipe):
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    con.update_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {"$push": {SAVED_RECIPES: {recipe: 0}}}
    )

    return f'Successfully added {recipe}'


def made_recipe(username, recipe):
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    con.update_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {"$inc": {f'{SAVED_RECIPES}.{recipe}': 1}}
    )

    return f'Successfully incremented streak counter for {recipe}'


def remove_recipe(username, recipe):
    con.connect_db()
    if not user_exists(username):
        raise ValueError(f'User {username} does not exist')
    if auth_expired(username):
        raise AuthTokenExpired("User's authentication token is expired")

    con.update_one(
        con.USERS_COLLECTION,
        {USERNAME: username},
        {"$pull": {SAVED_RECIPES: recipe}}
    )

    return f'Successfully removed {recipe}'


def recognize_receipt(image_path=None, image=None):
    openai.api_key = os.environ.get("OPENAI_KEY")
    if (image_path and not image):
        # Load the image from the specified path
        image = Image.open(image_path)
    elif (not image):  # neither the path nor image is provided
        return None
    # Perform OCR using pytesseract
    ocr_text = pytesseract.image_to_string(image)
    # Print or save the extracted text
    print(ocr_text)
    # Optionally, save the text to a file
    # with open('extracted_text.txt', 'w', encoding='utf-8') as file:
    #     file.write(text)

    # try:
    #     test = openai.api_key
    # except:
    #     return None
    # return ocr_text
    prompt = f"Extract pantry items from the following text: {ocr_text}"
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=200  # You can adjust this value based on your needs
    )
    # Extract the generated text from ChatGPT's response
    generated_text = response.choices[0].text.strip()
    # Split the generated text into individual pantry items
    pantry_items = generated_text.split('\n')
    # Remove any empty or whitespace-only items
    pantry_items = [item.strip() for item in pantry_items if item.strip()]
    return pantry_items
