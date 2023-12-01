__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import time
import numpy as np
from PyKakao import DaumSearch
from PyKakao import Local
import requests
from urllib import parse
import json
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import RetrievalQAWithSourcesChain
import os
import  streamlit_vertical_slider  as svs
import folium
from streamlit_folium import folium_static
import numpy as np
from streamlit_image_select import image_select
import streamlit_scrollable_textbox as stx
import chromadb
from langchain.vectorstores import Chroma
from streamlit_js_eval import streamlit_js_eval
import requests
import time
from streamlit_extras.switch_page_button import switch_page
import warnings
warnings.filterwarnings('ignore')
openai_key = st.secrets["openAI_key"]
openapi_key = st.secrets["openAPI_key"]
kakao_key = st.secrets["kakaO_key"]
os.environ['OPENAI_API_KEY'] = openai_key
LOCAL = Local(service_key = kakao_key)
DAUM = DaumSearch(service_key = kakao_key)
st.set_page_config(page_title="HashTrip",initial_sidebar_state="collapsed",layout="wide")
check_real = {0:12, 1:14, 2:15, 3:25, 4:28, 5:32, 6:38, 7:39}
real_name = {12:'관광지', 14:'문화시설', 15:'행사', 25:'여행코스', 28:'레포츠', 32:'숙박', 38:'쇼핑', 39:'음식점'}
embeddings = OpenAIEmbeddings()
state = st.session_state


if 'three_to_second' in state:
    del state.three_to_second
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


if "submitted" not in state:
    state.submitted = False
    state.refresh = 1
    state.button_sent = False
    state.go_back = False
    state.ans= {}
    state.data = {}
    state.pdf_data = {}
    state.data_check = {}
    state.hashtag = '' 

if state.go_back == True:
    state.go_back = False
    switch_page("page3")


def swich_to_next():
    state.go_back = True
    state.submitted = False
    streamlit_js_eval(js_expressions="parent.window.location.reload()")
    
def xy_search(df, setting_number = 10, search_number = 12):
    url = '	https://apis.data.go.kr/B551011/KorService1/locationBasedList1'
    queryParams = f'?{parse.quote_plus("serviceKey")}={openapi_key}&' + parse.urlencode({ 
                            parse.quote_plus('MobileOS') : 'ETC',
                            parse.quote_plus('MobileApp') : 'MobileApp',
                            parse.quote_plus('numOfRows') : setting_number,  
                            parse.quote_plus('contentTypeId') : search_number,
                            parse.quote_plus('_type') : 'json',                
                            parse.quote_plus('mapX') : df['x'][0],
                            parse.quote_plus('mapY') : df['y'][0],
                            parse.quote_plus('radius') : 10000})

    response = requests.get(url + queryParams)
    json_object = json.loads(response.text)
    if len(json_object['response']['body']['items']) > 0:
        return xy_json(json_object)
    else:
        return 0
    
def kakao_imagae(input_text):
    method = "GET"
    url = "https://dapi.kakao.com/v2/search/image"
    params = {'query' : input_text, 'page':1,  'size': 1}
    header = {'authorization': f'KakaoAK {kakao_key}'}
    response = requests.request(method=method, url=url, headers=header, params=params )
    tokens = response.json()
    return tokens['documents'][0]['image_url']

 
def kakao_blogs(input_text):
    method = "GET"
    url = "https://dapi.kakao.com/v2/search/blog"
    params = {'query' : input_text, 'page':1,  'size': 3}
    header = {'authorization': f'KakaoAK {kakao_key}'}
    response = requests.request(method=method, url=url, headers=header, params=params )
    tokens = response.json()
    return [tokens['documents'][i]['url'] for i in range(len(tokens['documents']))]

def xy_json(df_data):
    data_js = dict()
    addr = []
    img = []
    name = []
    mapx = []
    mapy = []
    blog_link = []

    for i in df_data['response']['body']['items']['item']:
        addr.append(i['addr1'])
        if len(i['firstimage']):
            img.append(i['firstimage'])
        else:
            img.append(kakao_imagae(i['title']))
        name.append(i['title'])
        mapx.append(i['mapx'])
        mapy.append(i['mapy'])
        blog_link.append(kakao_blogs(i['title']))


    # 혹시모를 빈값 방지
    data_js['addr'] =  [i  if i != '' else 'N'for i in addr]
    data_js['img'] = [i  if i != '' else 'N'for i in img]
    data_js['name'] = [i  if i != '' else 'N'for i in name]
    data_js['mapx'] = [i  if i != '' else 'N'for i in mapx]
    data_js['mapy'] = [i  if i != '' else 'N'for i in mapy]
    data_js['link'] = [i  if i != '' else 'N'for i in blog_link]
    # print(data_js)
    return data_js


def mk_db(df,sn):
    client = chromadb.PersistentClient(path=f"db{sn}")
    vec = []
    for n in range(len(df['addr'])):
        vec.append(Document(page_content= df['name'][n], metadata={'source': 'json', 'seq_num': n, 'addr': df['addr'][n], 'img': df['img'][n], 'mapx': df['mapx'][n],  'mapy': df['mapy'][n]}))
    
    globals()[f'vector_db{sn}'] = Chroma.from_documents(
                            client=client, 
                            documents=vec, 
                            embedding=embeddings, 
                            persist_directory=f'db{sn}'
                        )        
    
    return globals()[f'vector_db{sn}']


def QA_chatbot(_db, hash_str, rc ,op):
    system_template = """To answer the question at the end, use the following context. If you don't know the answer, just say you don't know and don't try to make it up.

    I want you to act as my hashtag travel recommendator. It tells you your hashtag, the number of recommendations, and suggests it according to the number of recommendations.

    You only answer in Korean
    {summaries}
    """
    messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    
    chain_type_kwargs = {"prompt": prompt}
    bk_chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=_db.as_retriever(search_kwargs={"k": rc}),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents = True
    )
    if op == 0:
        ans = bk_chain({"question": f'해시 태그 {hash_str}를 기반으로 {rc}개 추천설명을 수행해주세요.'})
        return  ans['answer'], ans['source_documents']
    else:
        ans = bk_chain({"question": f'해시 태그 {hash_str}를 기반으로 {rc}개 추천설명을 수행해주세요.'})
        make_ans_text = f'조회된 개수보다 추천 수가 많아서 조회된 개수 {rc}개 내에서 추천하는 것으로 변경되었습니다. \n\n ' +  ans['answer']
        return  make_ans_text, ans['source_documents']


def add_form(name, df, hash_str, rc, sn):
    count_brunch = len(df['addr'])
    brunch_n = 1

    if count_brunch == 0:
        brunch_n = 0
    elif rc > count_brunch:
        op = 1
        choice_ans, choice_doc= QA_chatbot(mk_db(df,sn), hash_str,count_brunch,op)
    elif rc <= count_brunch:
        op = 0
        choice_ans, choice_doc= QA_chatbot(mk_db(df,sn), hash_str,rc,op)
    
    if brunch_n:
        r_mapx = [float(i.metadata['mapx']) for i in choice_doc]
        r_mapy = [float(i.metadata['mapy']) for i in choice_doc]
        r_img = [i.metadata['img'] for i in choice_doc]
        r_name = [i.page_content for i in choice_doc]
        r_addr = [i.metadata['addr'] for i in choice_doc]
        r_num = [i.metadata['seq_num'] for i in choice_doc]
        
        state.pdf_data[name]= {'x' : r_mapx, 'y' : r_mapy, 'img' : r_img, 'name' : r_name, 'gpt_ans' : choice_ans, 'blog' : []}
        
        
        img_list = []
        for img_url in r_img:
            img_list.append({'width' : 500, 'height' : 500, 'src' : img_url})
                
        with st.form(f'{name}추천결과', clear_on_submit=True):
            change = st.form_submit_button(f'{name} 추천 결과',disabled=True)
            with st.expander(' '):
                c1, c2 = st.columns((30,70))
                empyt1,con4,empty2 = st.columns([0.1,1.0,0.1])
                c3,c4 = st.columns((1.0,0.1))
                c5,c6 = st.columns((1.0,0.1))
                empyt3,con5,empty4 = st.columns([0.1,1.0,0.1])
                
                m = folium.Map(location=[np.mean(r_mapy),np.mean(r_mapx)], zoom_start=13)
                
                for idx in range(len(r_num)):
                    folium.Marker(
                            [r_mapy[idx], r_mapx[idx]], popup=r_addr[idx], tooltip= r_name[idx]
                        ).add_to(m)
                        
                        
                with st.container():
                    with c1:
                        st_data = folium_static(m, width=300, height=300)
                    with c2:
                        choice_text = f"""
                        ChatGPT의 추천이유 입니다.
                        
                        {choice_ans}
                        """
                        stx.scrollableTextbox(choice_text, height=280,border=False,key=sn)
                    with con4:
                        st.divider()
                    with c3:
                        img_selected = image_select(
                            label="ChatGPT가 추천한 장소의 이미지 입니다.",
                            images=r_img,
                            captions=r_name,
                            )
                    with c5:
                        linked_text = ' '
                        for li_int in range(len(r_num)):
                            linked_text += f"[{li_int+1}.{r_name[li_int]}]({df['link'][r_num[li_int]][0]})"
                            state.pdf_data[name]['blog'].append(df['link'][r_num[li_int]][0])
                            linked_text += '  '
                        st.write(linked_text)
        return choice_ans, choice_doc, op
    else:
        st.write('조회 결과가 없습니다. ')
        

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }x
</style>
""",
    unsafe_allow_html=True,
)

button_css = st.markdown("""
<style>
div.stButton > button:first-child {

    position: relative;
    display: inline-block;
    font-size: px;
    color: white;

    border-radius: 6px;
    transition: top .01s linear;
    text-shadow: 0 1px 0 rgba(0,0,0,0.15);
    background-color: #82D1E3;
    border: none;
}
</style>

""", unsafe_allow_html=True)


left_column,  right_column = st.columns([25, 50])
_,  right_column_button = st.columns([25, 50])



with left_column:
        
    st.markdown('#### #️⃣ HashTrip 여행 선택지')
    st.write(' ')
    first_input = st.columns(2)
    with first_input[0]:
        st.markdown('- HashTrip #여행지 #여행테마 입력')
        hash_tags = st.text_input(
            "",
            label_visibility="collapsed",
            disabled=False,
            placeholder='#서울 #힐링 #여행',
        )
    with first_input[1]:
        st.markdown('- ChatGPT 항목 추천수를 입력')
        recomand_count = st.number_input(label = '', min_value=1, max_value=8,label_visibility="collapsed",value=3)
    if hash_tags:
        st.write(f"입력된 해시태그 : :red[**{hash_tags}**], ChatGPT 추천수 : :red[**{recomand_count}**]")
    
    
    st.write(' ')
    st.write(' ')
    st.markdown('#### ✅ HashTrip에 포함시킬 항목을 선택해 주세요. ')
    
    checks1 = st.columns(4)
    with checks1[0]:
        set1=st.checkbox('관광지')
    with checks1[1]:
        set2=st.checkbox('문화시설')
    with checks1[2]:
        set3=st.checkbox('행사')
    with checks1[3]:
        set4=st.checkbox('여행코스')
# ---------------------------
    checks2 = st.columns(4)
    with checks2[0]:
        set5 = st.checkbox('레포츠')
    with checks2[1]:
        set6 = st.checkbox('숙박')
    with checks2[2]:
        set7 = st.checkbox('쇼핑')
    with checks2[3]:
        set8 = st.checkbox('음식점')
        #set5,set6,set7,set8
    check_list = []
    for i in [set1,set2,set3,set4,set5,set6,set7,set8]:
        check_list.append(i)
        
    
    st.write('')
    st.write('')
    st.write(' ')
    
    min_value = recomand_count+1
    max_value = 12
    slid_number1,slid_number2,slid_number3,slid_number4 = 0,0,0,0
    slid_number5,slid_number6,slid_number7,slid_number8 = 0,0,0,0
    
    true_check_list = [idx for idx,i in enumerate(check_list) if i==True]
    num_col = check_list.count(True)  if check_list.count(True)  != 0 else 0
    st.markdown('#### 🤖 ChatGPT에게 전달할 항목 개수를 지정해 주세요.')
    if num_col:
        if num_col > 4:
            columns1 =  st.columns(4)
            columns2 = st.columns(num_col - 4)
            
            cnt = 0
            for idx, num in enumerate(true_check_list):
                if cnt >=4:
                    with columns2[idx-4]:
                        st.markdown(f'{real_name[check_real[num]]}')
                        globals()[f'slid_number{num+1}'] = svs.vertical_slider(key=f'set{num+1}', default_value=5, step=1, min_value=min_value, max_value=max_value)
                else:
                    with columns1[idx]:
                        st.markdown(f'{real_name[check_real[num]]}')
                        globals()[f'slid_number{num+1}'] = svs.vertical_slider(key=f'set{num+1}', default_value=5, step=1, min_value=min_value, max_value=max_value)               
                cnt +=1
        else:
            columns =  st.columns(num_col)
            for idx, num in enumerate(true_check_list):
                with columns[idx]:
                    st.markdown(f'{real_name[check_real[num]]}')
                    globals()[f'slid_number{num+1}'] = svs.vertical_slider(key=f'set{num+1}', default_value=5, step=1, min_value=min_value, max_value=max_value)
    
    st.write(f':red[**{len(true_check_list)}**]개 선택했습니다.')
    hash_list = [x.strip() for x in hash_tags.split('#') if x != '']

    left_btn, right_btn = st.columns([5,5])
    if len(true_check_list) >0:
        with left_btn:
            if state.button_sent:
                switch_page("page3")
            
            
            if st.button('Submit', key='button1'):
                gif_runner = st.image('hashloading.gif')
                time.sleep(0.5)
                for key_word in hash_list:
                    df_xy =  LOCAL.search_address(key_word, dataframe=True)
                    if len(df_xy):
                        state.hashtag = key_word
                        for idx,number in enumerate([check_real[i] for i in true_check_list]):
                            if globals()[f'slid_number{idx + 1}'] == None:
                                globals()[f'slid_number{idx + 1}']  = recomand_count+2
                            # st.write(df_xy)
                            globals()[f'df_{idx}'] = xy_search(df_xy,setting_number = globals()[f'slid_number{idx + 1}'], search_number = number)
                        hash_str = [i.strip() for i in hash_tags.split(f'#{key_word}') if len(i)>1]
                        if len(hash_str) == 1:
                            hash_str = hash_str[0]
                        else:
                            hash_str = ' '.join(hash_str)
                        break
                

    
    with right_column:
        placeholder = st.empty()
        with placeholder.container():
            name_list = [real_name[check_real[i]] for i in true_check_list]
            exist_list = []
            
            for i in range(8):
                if f'df_{i}' in globals(): 
                    exist_list.append(i)
                    # state.data[f'df_{i}'] = globals()[f'df_{i}']
            
            if len(exist_list) >=1 and state.refresh == 1:
                state.refresh += 1
                st.markdown('#### #️⃣ HashTrip 추천 항목')
                st.write(' ')
                for t in exist_list:
                    if f'df_{t}' in globals() and globals()[f'df_{t}'] != 0: 
                        set_name = name_list.pop(0)
                        # state.data[set_name] = globals()[f'df_{t}']
                        state.ans[set_name],  state.data[set_name], state.data_check[set_name] = add_form(set_name, globals()[f'df_{t}'], hash_str, recomand_count, t)
                    else:
                        st.write(f'{name_list.pop(0)} 조회 결과가 0 입니다. ')
                        st.write('다른 추천 항목을 선택해 주세요.')
                    time.sleep(0.5)
                
                gif_runner.empty()
                for i in exist_list:
                    if f'vector_db{i}' in globals(): 
                        # print(f'vector_db{i}  삭제중...')
                        for ids in globals()[f'vector_db{i}'].get()['ids']:
                            globals()[f'vector_db{i}'].delete(ids)
                        state.submitted = True


            if state.submitted:
            # if True:
                col1, col2, col3, col4, col5 = st.columns([5,4,3,2,0.6])
                
                with col1:
                    refresh_btn = st.button("Refresh",key='ref')
                    if refresh_btn:
                        state.submitted = False
                        streamlit_js_eval(js_expressions="parent.window.location.reload()")
            
                with col5:
                    next_page_btn = st.button("▶",key='npage', on_click=swich_to_next)