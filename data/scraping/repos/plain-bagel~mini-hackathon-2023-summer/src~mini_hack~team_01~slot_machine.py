# mypy: ignore-errors
import json
import os

import dotenv
import openai
import pandas as pd
import streamlit as st

from mini_hack.team_01.prompt_steps import prompt_step_1, prompt_step_2, prompt_step_3, prompt_step_4


# Load Environment variables and have them as constants
dotenv.load_dotenv()

# Get key from .env file
openai.api_key = os.getenv("OPENAI_API_KEY")


def send_chat(messages, stream=False, max_tokens=1_000):
    """
    Generate text using ChatGPT as a separate process
    """
    try:
        gpt_res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=1.0,
            max_tokens=max_tokens,
            stream=stream,
            n=1,
            messages=messages,
        )
    except Exception as e:
        print(e)
        return None

    return gpt_res["choices"][0]["message"]["content"]


@st.cache_data()
def show_actor_profile(user_input):
    """
    Step 1: Generate a bio of a character in the play.
    """
    messages = prompt_step_1.get_messages(user_input)
    response = send_chat(messages, max_tokens=500)
    try:
        print("Step 1 response")
        print(response)
        actor_profile_json = json.loads(response)
        current_profile = {
            "name": actor_profile_json["name"],
            "date_of_birth": actor_profile_json["date_of_birth"],
            "height": actor_profile_json["height"],
            "hobby": actor_profile_json["hobby"],
            "mbti": actor_profile_json["mbti"],
            "family_relations": actor_profile_json["family_relations"],
            "personality": actor_profile_json["personality"],
            "style": actor_profile_json["style"],
            "occupation": actor_profile_json["occupation"],
        }
        df = pd.DataFrame(
            [
                current_profile,
            ]
        )
        st.data_editor(df)
        return current_profile

    except Exception as e:
        print(e)
        return


def enable_button(button_key, enabled):
    st.session_state[button_key] = enabled


@st.cache_data
def show_actor_appearance(current_profile):
    """
    Step 2: Generate a detailed profile of the character.
    """
    st.subheader("캐릭터 외형 정보 보기")
    messages = prompt_step_2.get_messages(json.dumps(current_profile))
    response = send_chat(messages, max_tokens=1200)
    try:
        print("Step 2 response")
        print(response)
        actor_detail_json = json.loads(response)
        print(actor_detail_json)
        current_appearance = {
            "appearance": actor_detail_json["appearance"],
            "fashion": actor_detail_json["fashion"],
        }
        st.subheader("외모")
        st.write(actor_detail_json["appearance"])
        st.subheader("스타일")
        st.write(actor_detail_json["fashion"])
        return current_appearance
    except Exception as e:
        print(e)
        return {}


@st.cache_data
def show_actor_image(current_appearance):
    """
    Step 3. Generate an image of the character.
    """
    st.subheader("이미지 그려보기")
    messages = prompt_step_3.get_messages(json.dumps(current_appearance))
    response = send_chat(messages)
    try:
        print("Step 3 response")
        print(response)
        image_response = openai.Image.create(
            prompt=prompt_step_3.get_prompt(response.__str__()),
            n=1,
            size=f"{512}x{512}",
        )
        current_image_url = image_response["data"][0]["url"]
        st.image(current_image_url)
        return current_image_url

    except Exception as e:
        print(e)
        return {}


@st.cache_data
def show_behind_story(current_profile):
    """
    Step 4. Generate a behind story of the character.
    """
    st.subheader("캐릭터 생애 스토리 보기")
    messages = prompt_step_4.get_messages(json.dumps(current_profile))
    response = send_chat(messages, max_tokens=2700)
    try:
        print("Step 4 response")
        print(response)
        actor_detail_json = json.loads(response)
        current_history = {
            "happy_moment": actor_detail_json["happy_moment"],
            "tragic_moment": actor_detail_json["tragic_moment"],
            "current_life": actor_detail_json["current_life"],
        }

        st.subheader("소중한 기억")
        st.write(actor_detail_json["happy_moment"])

        st.subheader("트라우마")
        st.write(actor_detail_json["tragic_moment"])

        st.subheader("현재")
        st.write(actor_detail_json["current_life"])

        return current_history
    except Exception as e:
        print(e)
        return {}


def set_up_default_view():
    st.title("가상 캐스팅")
    user_input = st.text_input(
        label="캐릭터의 성격을 알려주세요.", placeholder="유튜버 겸 모델 출신 인플루언서 SNS없이 못 사는 인생은 폼생폼사 20대 초반 남자 캐릭터 만들어 줘."
    )
    if user_input.strip() == "":
        return

    current_profile = show_actor_profile(user_input)

    btn_history = st.button("캐릭터 생애 스토리 보기")
    if st.session_state.get("btn_history", False) is not True:
        st.session_state["btn_history"] = btn_history
    else:
        show_behind_story(current_profile)

    btn_appearance = st.button("외모 정보 보기")
    if st.session_state.get("btn_appearance", False) is not True:
        st.session_state["btn_appearance"] = btn_appearance
    else:
        current_appearance = show_actor_appearance(current_profile)
        btn_image = st.button("이미지 그려보기")

        if st.session_state.get("btn_image", False) is not True:
            st.session_state["btn_image"] = btn_image
        else:
            show_actor_image(current_appearance)


if __name__ == "__main__":
    set_up_default_view()
