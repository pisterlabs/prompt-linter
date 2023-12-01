from typing import List, Generator
from openai.openai_object import OpenAIObject

import streamlit as st
st.set_page_config(
    page_title="k_tranditional_drink",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import openai
import time
import os
openai.api_key = st.secrets.OPENAI_TOKEN
from supabase import create_client
import pickle
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from streamlit_extras.switch_page_button import switch_page

st.subheader("🍶", anchor="k_alcohol")
empty1, con1, empty2 = st.columns([0.3, 1.0, 0.3])
with empty1:
    st.empty()
with con1:
    st.image("./f_image/title_03.png")
    want_to_contribute = st.button("황금 카드를 뽑았다면?!🏠")
    if want_to_contribute:
        switch_page("home")
with empty2:
    st.empty()

@st.cache_resource(show_spinner=None)
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase_client = init_connection()

EMBEDDING_MODEL = "text-embedding-ada-002"


#STEP 2) 데이터 로드
@st.cache_resource(show_spinner=None, experimental_allow_widgets=True)
def load_data():
    feature_df = pd.read_csv("./data/feature_total_f.csv", encoding="utf-8")
    main_df = pd.read_csv("./data/main_total_no_features_f.csv", encoding="utf-8")
    ingredient_df = pd.read_csv("./data/ingredient_total_id_f.csv", encoding="utf-8")
    embedding_df = pd.read_csv("./data/embedding_f.csv", encoding="utf-8")
    emoji_df = pd.read_csv("./data/emoji_selected_f.csv", encoding="utf-8")
    food_df = pd.read_csv("./data/food_preprocessed_f.csv", encoding="utf-8-sig")
    return feature_df, main_df, ingredient_df, embedding_df, emoji_df, food_df

feature_df, main_df, ingredient_df, embedding_df, emoji_df, food_df = load_data()

@st.cache_resource(show_spinner=None, experimental_allow_widgets=True)
def embedding_c():
    embeddings = [np.array(eval(embedding)).astype(float) for embedding in embedding_df["embeddings"].values]
    stacked_embeddings = np.vstack(embeddings)

    return stacked_embeddings

stacked_embeddings = embedding_c()

#STEP 3) 캐시 불러오고 임베딩 저장하기
embedding_cache_path = "./data/recommendations_embeddings_cache.pkl"

try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

empty3, con2, empty4 = st.columns([0.3, 1.0, 0.3])
@st.cache_resource(show_spinner=None, experimental_allow_widgets=True)
def embedding_from_string(
    string: str,
    model: str = "text-embedding-ada-002",
    embedding_cache=embedding_cache
) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]

def generate_prompt(name, feature, situation_keyword, emotion_keyword):
    prompt = f"""
전통주 이름은 변경하지마세요.
전통주의 특징을 먼저 서술하세요.
그 다음, 상황 키워드와 감정 키워드를 넣어 전통주의 특징과 잘 어우러지게 추천 문구를 작성해 주세요.
공백을 포함하여 200자 미만으로 작성해 주세요.
구어체의 공손하고 친절한 존댓말로 작성해 주세요.

예시)
싱그러운 과일의 첫 맛과 바질로 마무리되는 끝 맛이 조화롭습니다. 
긴 겨울 끝 어느새 성큼 다가오는 따스한 봄처럼 상큼한 과실주로 절로 미소를 짓게 만듭니다.
축제, 파티, 그리고 기념일 같은 즐거운 시간을 더욱 풍성하게 채워줍니다.
가족과 친구, 그리고 연인들과 함께하는 소중한 순간을 기념하고 축하하는데 딱 어울리며, 선물로도 좋습니다.

연한 핑크빛 스위트 와인으로, 장미향이 은은하게 나는 달콤한 디저트와인입니다.
당도와 산도의 균형이 좋아 깔끔하고 단맛이 두드러지며, 주로 식전주나 식후주로 좋습니다.
레드 다이아몬드의 색과 부드러운 포도향이 매력적입니다.
떫은 맛, 타닌감, 산미는 적지만 잘 익은 포도의 맛 하나로 충분히 풍부한 맛을 느낄 수 있습니다.

---
전통주 이름: {name}
전통주 특징: {feature}
상황 키워드: {situation_keyword}
감정 키워드: {emotion_keyword}
---
"""
    return prompt

def request_chat_completion(prompt):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {"role": "system", "content": "당신은 글을 잘 쓰는 유능한 홍보 전문가입니다."},
        {"role": "user", "content": prompt}
    ],
    stream=True
)
    return response
    
def process_generated_text(streaming_resp: Generator[OpenAIObject, None, None]) -> str:
    report = []
    res_box = st.empty()
    for resp in streaming_resp:
        delta = resp.choices[0]["delta"]
        if "content" in delta:
            report.append(delta["content"])
            res_box.markdown("".join(report).strip())
        else:
            break
    result = "".join(report).strip()
    return result

@st.cache_resource(show_spinner=None, experimental_allow_widgets=True)
def get_idx_emoji(input_query, alcohol_min, alcohol_max):
    # 입력받은 쿼리 임베딩
    input_query_embedding = embedding_from_string(input_query, model=EMBEDDING_MODEL)

    # 임베딩 벡터간 거리 계산 (open ai 라이브러리 사용 - embeddings_utils.py)
    ## 도수 제한
    alcohol_limited_list = main_df.loc[
        (main_df["alcohol"] >= alcohol_min) & (main_df["alcohol"] <= alcohol_max)].index.tolist()
    source_embeddings = stacked_embeddings[alcohol_limited_list]

    distances = distances_from_embeddings(input_query_embedding, source_embeddings, distance_metric="cosine")

    # 가까운 벡터 인덱스 구하기 (open ai 라이브러리 사용 - embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    # 입력 받은 쿼리
    print(f"Query string: {input_query}")

    # k개의 가까운 벡터 인덱스 출력
    k_nearest_neighbors = 1
    k_counter = 0

    idx_list = []
    for i in indices_of_nearest_neighbors:
        # stop after printing out k articles
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1

        idx_list.append(i)

    return idx_list, alcohol_limited_list

def get_result(
        emotion: str,
        situation: str,
        ingredient: str,
        food: str,
        alcohol: str,
):

    if "\U0001F336" in ingredient or "\U0001F336" in food:
        ingredient = "\U0001F336"
        food = "\U0001F336"
    # query 수정
    situation_keyword = emoji_df[emoji_df["sample"] == situation]["k_keywords"].values[0]
    emotion_keyword = emoji_df[emoji_df["sample"] == emotion]["k_keywords"].values[0]
    ingredient_keyword = emoji_df[emoji_df["sample"] == ingredient]["k_keywords"].values[0]
    food_keyword = emoji_df[emoji_df["sample"] == food]["k_keywords"].values[0]

    input_query = f"재료는 {ingredient_keyword}다. 어울리는 음식으로는 {food_keyword}가 있다. {situation_keyword}다. {emotion_keyword} 감정을 언급할 수 있다."  # 벡터 임베딩용 쿼리
    result_query = f"{emotion} {situation} {ingredient} {food}"  # 출력용 쿼리

    # 알콜 이모지 도수로 변환
    if alcohol == "⬆️":
        alcohol_min = 18
        alcohol_max = 61

    else:
        alcohol_min = 0
        alcohol_max = 20

    idx_list, alcohol_limited_list = get_idx_emoji(input_query, alcohol_min, alcohol_max)

    name_id_list = []
    for i in idx_list:
        name_id_list.append(main_df.loc[alcohol_limited_list].iloc[i]["name_id"])

    # 결과 확인용
    print(f"{emotion}{situation}{food}로는 이게 딱!")

    for name_id in name_id_list:
        print(main_df[main_df["name_id"] == name_id]["name"].to_string(index=False))
        print(main_df[main_df["name_id"] == name_id]["alcohol"].to_string(index=False))
        print(feature_df[feature_df["name_id"] == name_id]["features"].to_string(index=False))
        print("---")

    return situation_keyword.split(",")[0], emotion_keyword.split(",")[0],  result_query, name_id

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


def image_name(name_id):
    directory = "./f_image/"
    matching_files = [file for file in os.listdir(directory) if name_id in file]
    if len(matching_files) > 0:
        filename = os.path.join(directory, matching_files[0])
        return filename  # 변수 filename을 반환합니다.
    else:
        return None


input_container = None

def write_propmt_result(emotion, situation, ingredient, food, name_id):
    supabase_client.table("result").insert(
        {
            "emotion": emotion,
            "situation": situation,
            "ingredient": ingredient,
            "food": food,
            "name_id": name_id,
        }
    ).execute()


with con2:
    container = st.empty()
    form = container.form("my_form", clear_on_submit=True)  # 내부 컨테이너의 폼 생성

    with form:
        empty7, col_s, empty9, col_e, empty8 = st.columns([0.05, 0.5, 0.2, 0.5, 0.05])
        with empty7:
            st.empty()

        with col_s:
            emotion = st.selectbox('감성', ('😁', '😭', '🥰', '😡', '😴', '🤢', '😱', '😎', '😂', '🥳'))

        with col_e:
            situation = st.selectbox("상황", ('☀️','☁️','❄️','🔥','☂️','💔','🎉','🎁','✈️','💍','💼','🚬','📝','💸','🌊','🌳','🍂','🌸','💪','👏','✌️','🙌','👍','👎'))

        with empty7:
            st.empty()

        empty10, col_i, empty15, col_f, empty11= st.columns([0.05, 0.5, 0.2, 0.5, 0.05])
        with empty10:
            st.empty()

        with col_i:
            ingredient = st.selectbox('재료', ('🍇','🍉','🍊','🍋','🍌','🍍','🍎','🍐','🍑','🍒','🍓','🍅','🌽','🌰','🥜',
                                             '🥔','🥕','🌶️','🍄','🌼','🎍','🌿','🍯','🥝','🥥','🌾','☕','🍵', '🍫','🍠','🧊','🥛'))

        with col_f:
            food = st.selectbox('어울리는 음식', ('🍕','🍔','🍟','🌭','🍿','🥞','🧈','🥐','🧀','🥗',
                                '🥩','🥟','🍤','🍱','🍚','🍜','🦪','🍣','🥘','🍝','🍦','🍩','🍪','🍰',
                                '🍫','🍬','🥛','🧃','🧊','🍯','🌶️','☕'))

        with empty11:
            st.empty()

        empty13, col_a, empty16, col_n, empty14= st.columns([0.05, 0.5, 0.2, 0.5, 0.05])
        with empty13:
            st.empty()
        with col_a:
            alcohol = st.selectbox('도수', ('⬇️','⬆️'))
        with empty16:
            st.empty()
        with col_n:
            real_name = st.text_input('이름 (선택)', placeholder="이름 또는 닉네임을 입력해주세요.")
        with empty14:
            st.empty()
        empty20,empty21,empty22,empty23,empty24,empty25 = st.columns(6)
        with empty25:
            submitted = st.form_submit_button("제출하기")

with st.container():  # 외부 컨테이너
    empty1, image_c, text_c, empty2 = st.columns([0.3, 0.3, 0.5, 0.3])
    name_id_list = []  # name_id_list 변수를 초기화합니다.
    if submitted:
        if not situation:
            st.error("어떤 상황에서 술을 마시고 싶은지 입력해주세요")
        elif not emotion:
            st.error("어떤 기분일 때 마시고 싶은지 입력해주세요")
        else:
            empty7, pro, empty9 = st.columns([0.3, 1.0, 0.3])
            with pro:
                with st.spinner('당신을 위한 전통주를 찾고 있습니다...🔍'):
                    situation_keyword, emotion_keyword, result_query, name_id = get_result(situation=situation, emotion=emotion, food=food,
                                                                    ingredient=ingredient, alcohol=alcohol)
                    write_propmt_result(emotion=emotion, situation=situation, ingredient=ingredient, food=food, name_id=name_id)
                    time.sleep(5)
                    if not name_id:
                        st.warning("검색 결과가 없습니다.")
                    else:
                        container.empty()
                        with image_c:
                            if name_id:
                                filtered_df = main_df[main_df["name_id"].str.contains(name_id)]
                                if not filtered_df.empty:
                                    loaded_image = image_name(name_id)
                                    st.image(loaded_image, use_column_width='auto')
                                else:
                                    st.write("해당하는 이미지가 없습니다.")

                        with text_c:
                            st.header(f"{emotion} {situation} {ingredient} {food}", anchor=False)
                            if real_name:
                                st.text(f"{real_name}님의 전통주 이모지 조합")
                            if name_id:
                                alcohol_name = main_df[main_df["name_id"]==name_id]["name"].to_string(index=False)
                                st.write(f"🔸 전통주 이름 : {alcohol_name}")
                                alcohol = main_df[main_df["name_id"] == name_id]["alcohol"].to_string(index=False)
                                st.write(f"🔸 도수 : {alcohol}")
                                st.write("🔸 특징 :")
                                features = feature_df[feature_df["name_id"] == name_id]["features"].to_string(index=False)
                                prompt = generate_prompt(name=alcohol_name, feature=features, situation_keyword=situation_keyword, emotion_keyword=emotion_keyword)
                                streaming_resp = request_chat_completion(prompt)
                                generated_text = process_generated_text(streaming_resp)
                                with_food = food_df[food_df["name_id"] == name_id]["food"].values[0]
                                st.write(f"🔸 어울리는 음식 : {with_food}")
                                if st.button('다시하기'):
                                    st.experimental_rerun()



                            else:
                                st.warning(f"전통주 이름: {name_id} 에 해당하는 정보가 없습니다.")





