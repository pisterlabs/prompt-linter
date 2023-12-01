# 載入套件
import openai
import streamlit as st

# 取得模型名稱
@st.cache_data
def get_model():
    response = openai.Model.list()
    model_list = []
    for model in response.data:    
        if model.id.startswith('gpt-3.5') or 'ft-' in model.id:
            model_list.append(model.id)
    
    return model_list

# 建立畫面
st.set_page_config(layout='wide')
st.title('ChatGPT 模型推論')
col1, col2 = st.columns((10, 2))
with col2:
    model = st.selectbox('模型', get_model()) 
    temperature = st.slider('temperature', 0.0, 1.0, 0.0, 0.1)
    max_tokens = st.slider('max_tokens', min_value=1, max_value=200, value=1)
    suffix = st.text_input('suffix', '->')
with col1:
    topic = st.text_input('topic')
    prompt = st.text_area('Prompt', value='', height=300)

    if st.button('執行'):
        # 呼叫 API 
        if model.startswith('gpt-3.5'):
            if len(topic) >= 1 and ord(topic[0]) >= 256: # 是否為中文
                system_prompt = f"You are a {topic} assistant"
            else:
                system_prompt = f"你是{topic}專家"
            
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt + suffix}
            ]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # 顯示回答
            st.text(response.choices[0].message.content)
        else: # 微調模型
            response = openai.Completion.create(
                model=model,
                prompt=prompt + suffix,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        
            # 顯示回答
            st.text(response.choices[0].text)