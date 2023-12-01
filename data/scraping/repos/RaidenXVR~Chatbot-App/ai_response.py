import openai
import json
import requests
import io
import time
import os
from pymongo import MongoClient
import pymongo
import bcrypt
import base64
from PIL import Image
import asyncio
import yaml
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import secrets
import certifi


def load_from_yaml(filename):
    with open(filename, "r") as file:
        data = yaml.safe_load(file)
    return data


def decrypt(encrypted_data, key):
    encrypted_data = base64.b64decode(encrypted_data)

    iv = encrypted_data[: AES.block_size]

    cipher = AES.new(key, AES.MODE_CBC, iv)

    decrypted_text = unpad(
        cipher.decrypt(encrypted_data[AES.block_size :]), AES.block_size
    )

    return decrypted_text.decode("utf-8")


# for normal chat
def get_response(input: str, user_id: int, topic: str) -> str:
    env = eval(
        decrypt(load_from_yaml("./assets/env.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    lsr = eval(
        decrypt(load_from_yaml("./assets/lsr.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    try:
        client = MongoClient(env["MONGODB"], tlsCAfile=certifi.where())
        db = client["Chatbot"]
        col = db["users"]
        user = col.find_one({"_id": user_id})
        message = [
            {
                "role": "system",
                "content": "You are an elite Teacher. And I am your friend whom you must pass on your knowledge and expertise. In a series of sessions, you have to fulfil this duty and help me answer my questions. Be friendly and informal as I am your friend. You can also generate images.",
            },
        ]
        last_message = []
        for msg in user["topics"]:
            if topic in msg.keys():
                message.extend(msg[str(topic)][:15])
                last_message = msg[str(topic)]

        message.append({"role": "user", "content": input})

        response, message_type = generate_response(message)

        if type(response) == str or message_type == "image":
            if user["num_valid_msg"] > 1:
                col.update_one(
                    {"_id": user_id},
                    {"$set": {"num_valid_msg": user["num_valid_msg"] - 2}},
                )
            elif user["num_valid_msg"] == 1 and not user["payed"]:
                return "Jumlah Batas Chat Anda Kurang Untuk Membuat Gambar.", "error"
            elif user["image_quota"] < 1 and user["payed"]:
                return "Jumlah Maksimal Pembuatan Gambar Perminggu Tercapai.", "error"
            elif user["image_quota"] > 0 and user["payed"]:
                col.update_one(
                    {"_id": user_id},
                    {"$set": {"image_quota": user["image_quota"] - 1}},
                )

            message = []
            message.append({"role": "user", "content": input})
            message.append(
                {"role": "assistant", "content": f"Generated Image Path: {response}"}
            )
            topics: list = user["topics"]
            topics.append({topic: message})
            col.update_one(
                {"_id": user_id},
                {
                    "$set": {
                        "topics": topics,
                    },
                },
            )
            return response, message_type, ""
        response = response.content

        message = []
        message.extend(last_message)
        message.append({"role": "user", "content": input})
        message.append({"role": "assistant", "content": response})

        topics: list = user["topics"]
        for tpc in topics:
            if topic in tpc.keys():
                tpc[topic] = message
        data = {"topics": topics}
        col.update_one({"_id": user_id}, {"$set": data})

        if user["num_valid_msg"] > 0:
            col.update_one(
                {"_id": user_id}, {"$set": {"num_valid_msg": user["num_valid_msg"] - 1}}
            )
        return message[-1]["content"], "chat", ""

    except openai.OpenAIError as e:
        return e.message, "error"
    except pymongo.errors.PyMongoError as e:
        return e._error_labels, "error"
    except Exception as e:
        return e._message, "error"


# for first new topic chat
def get_first_response(input: str, user_id):
    env = eval(
        decrypt(load_from_yaml("./assets/env.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    lsr = eval(
        decrypt(load_from_yaml("./assets/lsr.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    try:
        client = MongoClient(env["MONGODB"], tlsCAfile=certifi.where())
        db = client["Chatbot"]
        col = db["users"]
        user = col.find_one({"_id": user_id})
        message = []
        message.append(
            {
                "role": "system",
                "content": "Make a topic title from the user input. Send only the titles without any confirmations and don't use quotation marks or apostrophes. You should not make an image even if the user ask for it. Just make the topic only.",
            }
        )
        message.append({"role": "user", "content": input})
        try:
            topic, message_type = generate_topic(message)
            topic = topic.content

        except openai.OpenAIError as e:
            return e.message, "error"
        # Remove extra quotes and return the cleaned string
        if '"' in topic or "'" in topic:
            topic = topic.replace('"', "").replace("'", "")

        message = []
        message.append(
            {
                "role": "system",
                "content": "You are an elite Teacher. And I am your friend whom you must pass on your knowledge and expertise. In a series of sessions, you have to fulfil this duty and help me answer my questions. Be friendly and informal as I am your friend. Answer shortly, just answer or do as they say.",
            }
        )
        message.append({"role": "user", "content": input})

        response, message_type = generate_response(message)

        if message_type == "image":
            if user["num_valid_msg"] > 1:
                col.update_one(
                    {"_id": user_id},
                    {"$set": {"num_valid_msg": user["num_valid_msg"] - 2}},
                )
            elif user["num_valid_msg"] == 1 and not user["payed"]:
                return "Jumlah Batas Chat Anda Kurang Untuk Membuat Gambar.", "error"
            elif user["image_quota"] < 1 and user["payed"]:
                return "Jumlah Maksimal Pembuatan Gambar Perminggu Tercapai.", "error"
            elif user["image_quota"] > 0 and user["payed"]:
                col.update_one(
                    {"_id": user_id},
                    {"$set": {"image_quota": user["image_quota"] - 1}},
                )
            message = []
            message.append({"role": "user", "content": input})
            message.append(
                {"role": "assistant", "content": f"Generated Image Path: {response}"}
            )
            topics: list = user["topics"]

            tpc = {topic: message}
            topics.append(tpc)
            col.update_one(
                {"_id": user_id},
                {
                    "$set": {
                        "topics": topics,
                    },
                },
            )
            return response, message_type, topic
        else:
            message = []
            message.append({"role": "user", "content": input})
            message.append({"role": "assistant", "content": response.content})
            topics: list = user["topics"]
            topics.append({topic: message})
            col.update_one(
                {"_id": user_id},
                {
                    "$set": {
                        "topics": topics,
                    },
                },
            )
            if user["num_valid_msg"] > 0:
                col.update_one(
                    {"_id": user_id},
                    {"$set": {"num_valid_msg": user["num_valid_msg"] - 1}},
                )
            return response.content, topic, ""
    except openai._exceptions.OpenAIError as e:
        return f"{e.message}", "error"
    except openai.BadRequestError as e:
        return f"{e.message}", "error"
    except pymongo.errors.PyMongoError as e:
        return e._message, "error"
    except Exception as e:
        return e._message, "error"


def login(username: str, password: str):
    env = eval(
        decrypt(load_from_yaml("./assets/env.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    lsr = eval(
        decrypt(load_from_yaml("./assets/lsr.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    client = MongoClient(env["MONGODB"], tlsCAfile=certifi.where())
    db = client["Chatbot"]
    col = db["users"]
    user = col.find_one({"username": username})
    if user == None:
        return False  # signup/username not found
    elif bcrypt.checkpw(password.encode("utf-8"), user["p"]):
        data = {"USER_ID": user["_id"], "LOGIN": True}
        save_to_yaml(
            encrypt(str(data), b"ftrn80827310103rdnxvrnzr"), "./assets/lsr.yaml"
        )
        return True
    else:
        return False  # wrong password


def sign_up(email: str, username: str, password: str):
    env = eval(
        decrypt(load_from_yaml("./assets/env.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    lsr = eval(
        decrypt(load_from_yaml("./assets/lsr.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    try:
        client = MongoClient(env["MONGODB"], tlsCAfile=certifi.where())
        db = client["Chatbot"]
        col = db["users"]
        count = col.estimated_document_count()
        emails = col.find_one({"email": email})
        users = col.find_one({"username": username})
        if users != None or emails != None:
            print("User or Email Is Not None")
            return False  # username already in use

        hash_password = bcrypt.hashpw(
            password=password.encode("utf-8"), salt=bcrypt.gensalt()
        )
        user_id = generate_user_id()
        template = {
            "_id": user_id,
            "email": email,
            "username": username,
            "p": hash_password,
            "topics": [],
            "payed": False,
            "num_valid_msg": 0,
            "image_quota": 0,
        }

        col.insert_one(template)
        return True  # account signup succesful
    except pymongo.errors.PyMongoError as e:
        print("THE ERROR: " + str(e))

        return False


def logout():
    env = eval(
        decrypt(load_from_yaml("./assets/env.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    lsr = eval(
        decrypt(load_from_yaml("./assets/lsr.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    data = {"USER_ID": 0, "LOGIN": False}
    save_to_yaml(encrypt(str(data), b"ftrn80827310103rdnxvrnzr"), "./assets/lsr.yaml")
    return True


def get_profile(user_id: int):
    try:
        env = eval(
            decrypt(load_from_yaml("./assets/env.yaml"), b"ftrn80827310103rdnxvrnzr")
        )
        lsr = eval(
            decrypt(load_from_yaml("./assets/lsr.yaml"), b"ftrn80827310103rdnxvrnzr")
        )
        client = MongoClient(env["MONGODB"], tlsCAfile=certifi.where())
        db = client["Chatbot"]
        col = db["users"]
        user = col.find_one({"_id": user_id})
        return user["username"], user["email"]

    except pymongo.errors.PyMongoError as e:
        return e
    except Exception as e:
        return e


def generate_response(messages: list = None, prompt: str = None) -> str:
    env = eval(
        decrypt(load_from_yaml("./assets/env.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    lsr = eval(
        decrypt(load_from_yaml("./assets/lsr.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "a function to generate or make an image using a given prompt. Used when user ask to make or generate an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "prompt for generating the image, e.g. A professionally taken Photograph of a cat on a window.",
                        }
                    },
                },
            },
        }
    ]
    openai_client = openai.OpenAI(api_key=env["OPENAI_API"])
    if not prompt == None:
        messages = [{"role": "user", "content": prompt}]
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-1106-preview", messages=messages, tools=tools
        )
        if (
            response.choices[0].message.content == None
            or response.choices[0].message.tool_calls != None
        ):
            client = MongoClient(env["MONGODB"], tlsCAfile=certifi.where())
            db = client["Chatbot"]
            col = db["users"]
            user = col.find_one({"_id": lsr["USER_ID"]})
            if user["num_valid_msg"] < 1 and not user["payed"]:
                return "", "image"
            param = response.choices[0].message.tool_calls[0].function.arguments
            function = eval(response.choices[0].message.tool_calls[0].function.name)
            try:
                image_path = function(param)
            except openai.OpenAIError as e:
                return e
            except openai.BadRequestError as e:
                return e
            return image_path, "image"

        return response.choices[0].message, "chat"
    except openai.OpenAIError as e:
        return e


def generate_topic(messages: list = None, prompt: str = None) -> str:
    env = eval(
        decrypt(load_from_yaml("./assets/env.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    lsr = eval(
        decrypt(load_from_yaml("./assets/lsr.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    openai_client = openai.OpenAI(api_key=env["OPENAI_API"])
    if not prompt == None:
        messages = [{"role": "user", "content": prompt}]
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-1106-preview", messages=messages
        )
        return response.choices[0].message, "topic"
    except openai.OpenAIError as e:
        return e


def generate_image(prompt: str):
    env = eval(
        decrypt(load_from_yaml("./assets/env.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    lsr = eval(
        decrypt(load_from_yaml("./assets/lsr.yaml"), b"ftrn80827310103rdnxvrnzr")
    )
    openai.api_key = env["OPENAI_API"]

    prompt = eval(prompt)["prompt"]
    try:
        response = openai.images.generate(
            prompt=prompt,
            model="dall-e-3",
            quality="standard",
            response_format="b64_json",
            size="1024x1024",
            n=1,
        )
    except openai.OpenAIError as e:
        return e
    tm = time.strftime("%d%m%Y-%H.%M%S")
    image64 = response.data[0].b64_json

    image_data = base64.b64decode(image64)
    image = Image.open(io.BytesIO(image_data))
    if not os.path.exists("./img"):
        os.mkdir("./img")
    image.save(f"./img/{tm}.png")
    return f"./img/{tm}.png"


def test_input(input: str, type: str):
    time.sleep(3)

    return input, type


def encrypt(text, key):
    iv = get_random_bytes(AES.block_size)

    cipher = AES.new(key, AES.MODE_CBC, iv)

    padded_text = pad(text.encode("utf-8"), AES.block_size)

    ciphertext = cipher.encrypt(padded_text)

    encrypted_data = base64.b64encode(iv + ciphertext)

    return encrypted_data.decode("utf-8")


def save_to_yaml(data, filename):
    with open(filename, "w") as file:
        yaml.dump(data, file)


def generate_user_id():
    random_bytes = secrets.token_bytes(4)

    user_id = int.from_bytes(random_bytes, byteorder="big", signed=False)

    user_id = user_id % 90000000 + 10000000
    return user_id
