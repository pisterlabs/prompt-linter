import redis
from redis.commands.search.query import NumericFilter, Query
from redis.commands.json.path import Path
from credentials import REDIS_URL, REDIS_PASSWORD, SUPABASE_KEY, SUPABASE_URL, OPENAI_KEY
from supabase import create_client, Client
from random import choice, sample
import json

import openai

openai.api_key = OPENAI_KEY

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

r = redis.Redis(
  host=REDIS_URL,
  port=17728,
  password=REDIS_PASSWORD
)

def is_onboarded(email) -> [int]:
    try:
        response = supabase.table("topic_selection").select("*").eq('email', email).execute()
        if len(response.data) > 0:
            return response.data[0]["selection_list"]
        return []
    except:
        return []

def onboard(email, topic_selection):
    supabase.table("topic_selection").insert({"email": email, "selection_list": topic_selection}).execute()

    for i in topic_selection:
        key = f"current:{email}:{i}"

        data = {"unit": 1, "reel": 1}
        r.json().set(key, Path.root_path(), json.dumps(data))

def increment_current(reel_id, email):
    reel = supabase.table("reels").select("*").eq("id", reel_id).limit(1).single().execute().data
    unit_id = reel["unit_id"]
    unit = supabase.table("units").select("*").eq("id", unit_id).limit(1).single().execute().data
    topic_id = unit["topic_id"]
    response = supabase.table("reels").select("*").filter("number", "eq", reel["number"] + 1).filter("unit_id", "eq", unit_id).limit(1).execute()
    if len(response.data) > 0:
        key = f"current:{email}:{topic_id}"
        data = json.loads(r.json().get(key))
        data["reel"] = reel["number"] + 1
        r.json().set(key, Path.root_path(), json.dumps(data))
        return
    
    response = supabase.table("units").select("*").filter("number", "eq", unit["number"] + 1).filter("topic_id", "eq", topic_id).limit(1).execute()

    if len(response.data) > 0:
        key = f"current:{email}:{topic_id}"
        data = json.loads(r.json().get(key))
        data["reel"] = 1
        data["unit"] = unit["number"] + 1
        r.json().set(key, Path.root_path(), json.dumps(data))
    else:
        key = f"current:{email}:{topic_id}"
        data = {"unit": "finished", "reel": "finished"}
        r.json().set(key, Path.root_path(), json.dumps(data))

def get_meme_reel():
    response = supabase.table("meme_reels").select("*").execute()
    return choice(response.data)

def recommend_reel(email, selection_list):
    data_list = []

    for i in selection_list:
        data = json.loads(r.json().get(f"current:{email}:{i}"))
        if data["unit"] != "finished":
            data_list.append({"data": data, "topic": i})

    print(data_list)
    k = choice(data_list)
    
    try:
        unit = supabase.table("units").select("*").filter("topic_id", "eq", k["topic"]).filter("number", "eq", k["data"]["unit"]).limit(1).single().execute().data
        topic = supabase.table("topics").select("*").filter("id", "eq", unit["topic_id"]).limit(1).single().execute().data
        reel = supabase.table("reels").select("*").filter("unit_id", "eq", unit["id"]).filter("number", "eq", k["data"]["reel"]).limit(1).single().execute().data
        reel["unit"] = unit["name"]
        reel["topic"] = topic["name"]

        return reel
    except:
        pass
    
    return get_meme_reel()

def choose_two_elements(lst):
    if len(lst) < 2:
        return lst
    else:
        return sample(lst, 2)
    
def recommend_initial(email, selection_list):
    final = []
    data_list = []

    for i in selection_list:
        data = json.loads(r.json().get(f"current:{email}:{i}"))
        if data["unit"] != "finished":
            data_list.append({"data": data, "topic": i})
    kk = choose_two_elements(data_list)
    
    for k in kk:
        try:
            unit = supabase.table("units").select("*").filter("topic_id", "eq", k["topic"]).filter("number", "eq", k["data"]["unit"]).limit(1).single().execute().data
            topic = supabase.table("topics").select("*").filter("id", "eq", unit["topic_id"]).limit(1).single().execute().data
            reel = supabase.table("reels").select("*").filter("unit_id", "eq", unit["id"]).filter("number", "eq", k["data"]["reel"]).limit(1).single().execute().data

            reel["unit"] = unit["name"]
            reel["topic"] = topic["name"]
            final.append(reel)
        except:
            pass
    
    final += get_n_meme_reels(3 - len(final))
    return final

def get_n_meme_reels(n):
    response = supabase.table("meme_reels").select("*").execute()
    return sample(response.data, n)


def offboard():
    supabase.table("topic_selection").delete().neq("email", 1).execute()

def get_comments(reel_id):
    comments = []
    cursor = '0'

    while cursor != 0:
        cursor, keys = r.scan(cursor, match=f"comments:{reel_id}:*")
        for key in keys:
            comment_data = r.json().get(key)
            if comment_data:
                comments.append(json.loads(comment_data))

    return comments

def remove_comments(reel_id):
    cursor = '0'

    while cursor != 0:
        cursor, keys = r.scan(cursor, match=f"comments:{reel_id}:*")
        for key in keys:
            r.delete(key)

def add_comment(reel_id, email, name, profile_url, created_at, comment):
    key = f"comments:{reel_id}:{email}:{created_at}"
    comment_data = {
        "email": email,
        "name": name,
        "profile_url": profile_url,
        "reel_id": reel_id,
        "content": comment,
        "created_at": created_at,
        "edison_reply": ""
    }

    r.json().set(key, Path.root_path(), json.dumps(comment_data))
    return comment_data

def chatgpt(prompt):
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def get_edison_reply(reel_id, content):
    response = supabase.table("reels").select("transcript").eq("id", int(reel_id)).limit(1).single().execute()
    transcript = response.data['transcript']

    prompt = f'based on an educational video with the following transcript: {transcript}\n\n answer the following question in a concise manner: {content}'
    return chatgpt(prompt)

def add_comment_with_edison(reel_id, email, name, profile_url, created_at, comment):
    key = f"comments:{reel_id}:{email}:{created_at}"
    comment_data = {
        "email": email,
        "name": name,
        "profile_url": profile_url,
        "reel_id": reel_id,
        "content": comment,
        "created_at": created_at,
        "edison_reply": get_edison_reply(reel_id, comment)
    }

    r.json().set(key, Path.root_path(), json.dumps(comment_data))
    return comment_data

def is_liked(reel_id, email) -> bool:
    key = f"likes:{reel_id}:{email}"
    return bool(r.exists(key))

def like_reel(reel_id, email):
    key = f"likes:{reel_id}:{email}"

    like_data = {
        "email": email,
        "reel_id": reel_id
    }

    r.json().set(key, Path.root_path(), json.dumps(like_data))
    increment_current(reel_id, email)

def remove_like(reel_id, email):
    key = f"likes:{reel_id}:{email}"

    r.delete(key)

# r.flushall()
# offboard()
# onboard("nand.vinchhi@gmail.com", [1, 2, 4])
# print(recommend_initial("nand.vinchhi@gmail.com", [1, 2, 4]))
# print(r.json().get("current:nand.vinchhi@gmail.com:1"))
# like_reel(5, "nand.vinchhi@gmail.com")
# print(r.json().get("current:nand.vinchhi@gmail.com:1"))


# print(r.json().get("current:nand.vinchhi@gmail.com:4"))