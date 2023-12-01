import streamlit as st
import openai
from supabase import create_client

@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase_client = init_connection()

st.markdown(
    """
<style>
footer {
    visibility: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


openai.api_key = st.secrets.OPENAI_TOKEN
openai_model_version = "gpt-3.5-turbo"

st.title("🎶 AI Lyrics")
st.text("AI를 이용하여 나만의 가사를 생성해 보세요.")




def generate_prompt(genre, theme, vibe, gender, keywords):
    if genre == "K-POP":
        prompt = f"""
        {theme}을 주제로 {vibe} 분위기의 노래 가사를 생성해 주세요.
        반드시 특정한 단어 및 구절이나 음운 패턴의 반복이 있어야 합니다.
        K-POP의 특징이 드러나야 합니다.
        키워드가 주어질 경우, 반드시 키워드가 포함된 구(Phrase)를 포함해야 합니다.
        성별이 주어질 경우, 성별이 말하는 듯이 작성해 주세요.
        주제와 관련있는 서사를 가사에 적용해 주세요.
        한국어로 가사를 만들되 3개 이상의 영어 단어를 포함해 주세요.
        아래의 예시는 참고용이므로 인용하지 마세요.
        
        예시)
        왜 자꾸 그녀만 맴도나요
        달처럼 그대를 도는 내가 있는데
        한 발짝 다가서지 못하는
        이런 맘 그대도 똑같잖아요
        오늘도 그녀 꿈을 꾸나요
        그댈 비춰주는 내가 있는데
        그렇게 그대의 하룬 또 끝나죠
        내겐 하루가 꼭 한 달 같은데
        그 꿈이 깨지길 이 밤을 깨우길
        잔잔한 그대
        그 마음에 파도가 치길
        너는 내 Destiny
        날 끄는 Gravity
        고개를 돌릴 수가 없어
        난 너만 보잖아
        너는 내 Destiny
        떠날 수 없어 난
        넌 나의 지구야
        내 하루의 중심
        
        ---
        노래 장르: {genre}
        주제: {theme}
        분위기: {vibe}
        성별: {gender}
        키워드: {keywords}
        ---
"""

    elif genre == "Hip hop":
        prompt = f"""
        {theme}을 주제로 {vibe} 분위기의 노래 가사를 생성해 주세요.
        반드시 특정한 단어 및 구절이나 음운 패턴의 반복이 있어야 합니다.
        힙합의 특징이 드러나야 합니다.
        키워드가 주어질 경우, 반드시 키워드가 포함된 구(Phrase)를 포함해야 합니다.
        성별이 주어질 경우, 성별이 말하는 듯이 작성해 주세요.
        주제와 관련있는 서사를 가사에 적용해 주세요.
        한국어로 가사를 만들되 영어 단어를 많이 사용해 주세요.
        아래의 예시는 참고용이므로 인용하지 마세요.
        
        예시)
        지난 날들이 모여
        지금 앞이 안 보여
        스쳐 지나가듯이 하지 마 별거 아닌 듯이
        날 담은 눈은 이미 투명해 전부 잊은 듯이
        깊게 나를 차지하는 널 이제 꺼내려 해봐도 replay
        It's hard to forget your big shade my lover
        깊게 너를 차지하는 건 이젠 내가 아니어도 keep waiting
        Or maybe just blaming
        What is it that you really wanna say
        I cannot get this over
        Smoking all the pain to feel you deeply again

        ---
        노래 장르: {genre}
        주제: {theme}
        분위기: {vibe}
        성별: {gender}
        키워드: {keywords}
        ---
"""
    elif genre == "R&B":
        prompt = f"""
        {theme}을 주제로 {vibe} 분위기의 노래 가사를 생성해 주세요.
        반드시 특정한 단어 및 구절이나 음운 패턴의 반복이 있어야 합니다.
        R&B의 특징이 드러나야 합니다.
        키워드가 주어질 경우, 반드시 키워드가 포함된 구(Phrase)를 포함해야 합니다.
        성별이 주어질 경우, 성별이 말하는 듯이 작성해 주세요.
        주제와 관련있는 서사를 가사에 적용해 주세요.
        한국어로 가사를 만들되 영어 단어를 많이 사용해 주세요.
        아래의 예시는 참고용이므로 인용하지 마세요.

        예시)
        내게 언젠가 왔던 너의 얼굴을 기억해
        멈춰있던 내 맘을 밉게도 고장난 내 가슴을
        너의 환한 미소가 쉽게도 연 거야
        그래 그렇게 내가 너의 사람이 된 거야
        못났던 내 추억들이 이젠 기억조차 않나
        나를 꼭 잡은 손이 봄처럼 따뜻해서
        이제 꿈처럼 내 맘은 그대 곁에 가만히 멈춰서요
        한순간도 깨지 않는 끝없는 꿈을 꿔요
        이제 숨처럼 내 곁에 항상 쉬며 그렇게 있어주면
        Nothing better
        Nothing better than you
        Nothing better
        Nothing better than you

        ---
        노래 장르: {genre}
        주제: {theme}
        분위기: {vibe}
        성별: {gender}
        키워드: {keywords}
        ---
        """
    else:
        prompt = f"""
        {theme}을 주제로 {vibe} 분위기의 노래 가사를 생성해 주세요.
        반드시 특정한 단어 및 구절이나 음운 패턴의 반복이 있어야 합니다.
        발라드의 특징이 드러나야 합니다.
        키워드가 주어질 경우, 반드시 키워드가 포함된 구(Phrase)를 포함해야 합니다.
        성별이 주어질 경우, 성별이 말하는 듯이 작성해 주세요.
        한국어로 가사를 만들되 영어 단어를 많이 사용해 주세요.
        아래의 예시는 참고용이므로 인용하지 마세요.

        예시)
        빛이 들어오면 자연스레 뜨던 눈
        그렇게 너의 눈빛을 보곤
        사랑에 눈을 떴어
        항상 알고 있던 것들도 어딘가
        새롭게 바뀐 것 같아
        남의 얘기 같던 설레는 일들이
        내게 일어나고 있어
        나에게만 준비된 선물 같아
        자그마한 모든 게 커져만 가
        항상 평범했던 일상도
        특별해지는 이 순간
        별생각 없이 지나치던 것들이
        이제는 마냥 내겐 예뻐 보이고
        내 맘을 설레게 해
        항상 어두웠던 것들도 어딘가
        빛나고 있는 것 같아

        ---
        노래 장르: {genre}
        주제: {theme}
        분위기: {vibe}
        성별: {gender}
        키워드: {keywords}
        ---
        """
    return prompt.strip()

def request_chat_completion(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 유용한 도우미입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]


def write_propmt_result(genre, theme, vibe, gender, result, keywords):
    #테이블에 새로운 레코드 삽입
    response = supabase_client.table("prompt_results").insert(
        {
            "genre": genre,
            "theme": theme,
            "vibe": vibe,
            "gender": gender,
            "result": result,
        }
    ).execute()

    result_id = response.data[0]['result_id']

    if keywords:
        for keyword in keywords:
            supabase_client.table("keywords").insert(
                {
                    "result_id": result_id,
                    "keywords": keyword,
                }
            ).execute()


genre_list = ['K-POP', 'Hip hop', '발라드', 'R&B']
gender_list = ['남성', '여성', '성별 무관']

with st.form("my_form"):
    theme = st.text_input("주제", placeholder="첫사랑 / 이별 / 자기 자랑 등")
    genre = st.selectbox("장르", genre_list)
    vibe = st.text_input("곡의 분위기", placeholder="잔잔한 / 신나는 / 조용한 등")
    gender = st.selectbox("가수 성별", gender_list)
    st.text("포함할 키워드(선택)")

    col1, col2, col3 = st.columns(3)
    with col1:
        keyword_one = st.text_input(
            placeholder="키워드 1",
            label="keyword_one",
            label_visibility="collapsed"
        )

    with col2:
        keyword_two = st.text_input(
            placeholder="키워드 2",
            label="keyword_two",
            label_visibility="collapsed"
        )

    with col3:
        keyword_three = st.text_input(
            placeholder="키워드 3",
            label="keyword_three",
            label_visibility="collapsed"
        )
    submitted = st.form_submit_button("Submit")

    if submitted:
        if not genre:
            st.error("장르를 선택해 주세요.")
        if not theme:
            st.error("주제를 입력해 주세요.")
        if not gender:
            st.error("성별을 선택해 주세요.")
        if not vibe:
            st.error("곡의 분위기를 입력해 주세요.")
        else:
            with st.spinner("AI 작사가가 가사를 생성하는 중입니다."):
                keywords = [keyword_one, keyword_two, keyword_three]
                keywords = [x for x in keywords if x]
                prompt = generate_prompt(genre, theme, vibe, gender, keywords)
                result = request_chat_completion(prompt)
                write_propmt_result(genre, theme, vibe, gender, result, keywords)

                st.text_area(
                    label="가사 생성 결과",
                    value=result,
                    height=200,
                )

st.text(f"Powerd by {openai_model_version}")