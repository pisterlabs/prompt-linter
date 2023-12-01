# 載入套件
import openai
import streamlit as st

# 取得模型名稱
@st.cache_data
def get_model():
    response = openai.Model.list()
    model_list = []
    for model in response.data:    
        if 'ft-' in model.id:
            model_list.append(model.id)
    
    return model_list

# 建立畫面
st.title('ChatGPT 模型推論')
model = st.selectbox('模型', get_model())    
prompt = st.text_input('Prompt') + '->'
if st.button('執行'):
    # 呼叫 API    
    max_tokens = 100
    if 'ft-' in model:
        max_tokens = 1
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0,
        max_tokens=max_tokens,
    )
    
    # 顯示回答
    st.text(response.choices[0].text)