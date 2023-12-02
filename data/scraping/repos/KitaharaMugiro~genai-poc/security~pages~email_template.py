import openai
import streamlit as st
import re


openai.api_base = "https://oai.langcore.org/v1"
# メールアドレスの正規表現
EMAIL_REGEX = r"[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}"

def replace_email_with_temp(text):
    found_emails = re.findall(EMAIL_REGEX, text)
    if not found_emails:
        return text, None
    # 入力テキスト内のメールアドレスをtemp@test.comに置換
    replaced_text = re.sub(EMAIL_REGEX, "temp@test.com", text)
    # 最初に見つかったメールアドレスを返す
    return replaced_text, found_emails[0]

def revert_email_in_text(text, original_email):
    if original_email:
        text = text.replace("temp@test.com", original_email)
    return text

st.title("メールアドレス差し替え")
st.write("個人情報であるメールアドレスをOpenAIに送る前にマスクし、メール文章生成時にアンマスクする")

# ユーザからの入力
placeholder_text = "私のメールアドレスは、mymail@langcore.orgです。このメールアドレスを使って文章を作成して。"
user_input = st.text_area("プロンプト:", value=placeholder_text)
button = st.button("OpenAIに送信")

if button:
    modified_input, original_email = replace_email_with_temp(user_input)
    
    st.subheader("OpenAIに送られるプロンプト:")
    st.code(modified_input)

    # OpenAIで文章生成
    with st.spinner("Thinking..."):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages= [
                {
                    "role": "user",
                    "content": modified_input
                }
            ]
        )
        generated_text = response.choices[0].message.content
        
        # メールアドレスを元に戻す
        final_text = revert_email_in_text(generated_text, original_email)
        
    st.subheader("生成された文章:")
    st.code(final_text)

