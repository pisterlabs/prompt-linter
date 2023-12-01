#coding part
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import pickle
import os
#load api key lib
from dotenv import load_dotenv
import base64
import time

from wanted import create_wanted_template_v2

def get_career_info(vectorstore):
    career_info = []
    
    has_career = get_response_from_predefined_query(vectorstore, "작성자는 회사 경력이 있는지에 대해 1(있다), 0(없다)로 답하라.")

    if has_career == "0":
        return "경력이 없음"

    companies = get_response_from_predefined_query(vectorstore, "작성자가 다녔던 회사는 어디인가?(단답형, 쉼표로 구분,회사명만 출력, 예시 : 삼성전자, 네이버)")
    
    companies = companies.split(', ')
    
    for company in companies:
        company_info = {}
        company_info["company"] = company

        position = get_response_from_predefined_query(vectorstore, f"{company}에서의 직무는?(단답형, 예시 : 프론트엔드 개발자)")
        company_info["position"] = position if position else "N/A"
        
        duration = get_response_from_predefined_query(vectorstore, f"{company}에서의 기간은?(단답형, 기간만을 출력하여라, 예시 : 2017.03 ~ 2022.02)")
        company_info["duration"] = duration if duration else "N/A"
        
        projects = get_response_from_predefined_query(vectorstore, f"{company}에서의 진행한 프로젝트명은?(단답형, 쉼표로 구분, 프로젝트 명 만을 출력하라,예시 : Galaxy S20 UI 개발, 네이버 메인페이지 리뉴얼)")
        project_list = []
        
        if projects:
            projects = projects.split(', ')
            
            for project in projects:
                project_info = {}
                project_info["name"] = project

                details = get_response_from_predefined_query(vectorstore, f"{project}의 세부내용은?(100타 이내로 미괄식 작성)")
                project_info["details"] = details if details else "N/A"

                project_duration = get_response_from_predefined_query(vectorstore, f"{project}의 기간은?(단답형, 기간만을 출력하여라, 예시 : 2017.03 ~ 2022.02)")
                project_info["duration"] = project_duration if project_duration else "N/A"
                
                project_list.append(project_info)
        
        company_info["projects"] = project_list
        career_info.append(company_info)
        
    return career_info

def get_education_info(vectorstore):
    education_info = []
    
    schools = get_response_from_predefined_query(vectorstore, "작성자가 다녔던 학교명은?(단답형, 쉼표로 구분, 예시 : 동의대학교)")
    
    if not schools:
        return "학력 정보가 없습니다."
        
    schools = schools.split(', ')
    
    for school in schools:
        school_info = {}
        school_info['school_name'] = school

        major = get_response_from_predefined_query(vectorstore, f"{school}에서의 전공은?(단답형, 예시 : 로봇자동화공학)")
        school_info['major'] = major if major else "N/A"
        
        duration = get_response_from_predefined_query(vectorstore, f"{school}을 다닌 기간은?(단답형, 기간만을 출력하여라, 예시 : 2017.03 ~ 2022.02)")
        school_info['duration'] = duration if duration else "N/A"
        
        education_info.append(school_info)
        
    return education_info

def get_skills_list(vectorstore):
    skills_query = "사용자의 기술스택(skill)을 나열하라.(단답형, 쉼표로 구분, 예시 : Python, C++, Java)"
    skills_response = get_response_from_predefined_query(vectorstore, skills_query)
    
    if not skills_response or skills_response.lower() == 'none':
        return ["기술 스택 정보가 없습니다."]
        
    skills_list = skills_response.split(', ')
    
    # 최대 10개의 스킬만 저장
    if len(skills_list) > 10:
        skills_list = skills_list[:10]
    
    return skills_list

def get_awards_and_others(vectorstore):
    awards_query = "작성자의 수상한 상 또는 수료한 교육은?(단답형, 쉼표로 구분, 예시 : 디지털스마트아카데미, 대통령상)"
    awards_response = get_response_from_predefined_query(vectorstore, awards_query)
    
    if not awards_response or awards_response.lower() == 'none':
        return "수상 및 교육 수료 정보가 없습니다."
        
    awards_list = awards_response.split(', ')
    awards_and_others = []
    
    for award in awards_list:
        date_query = f"{award}의 수료 날짜를 출력하라?(단답형, 일자만을 출력 하라, 예시 : 2017.02)"
        date_response = get_response_from_predefined_query(vectorstore, date_query)
        
        details_query = f"{award}의 세부 내용을 출력하라?(없을경우 공백 처리, 100타 이내로 미괄식 작성)"
        details_response = get_response_from_predefined_query(vectorstore, details_query)
        
        awards_and_others.append({
            "type": award,
            "details": details_response,
            "date": date_response
        })
    
    return awards_and_others

def get_introduce(vectorstore):
    skill_query = "작성자는 무엇을 할 줄 아는가?(작성자의 입장에서의 대답을 출력하라 ,예시 : 저는 프론트엔드 개발을 할줄아는 홍길동 입니다.)"
    interest_query = "작성자는 무엇에 관심 있는가? (작성자의 입장에서의 대답을 출력하라 , 예시 : 웹서비스 개발에 관심이 많습니다.)"
    career_query = "작성자는 어떠한 경력이 있는가?(마크다운으로 작성, 예시 : 프로젝트 리딩 및 다양한 팀과 협업 경험)"
    
    skill_response = get_response_from_predefined_query(vectorstore, skill_query)
    interest_response = get_response_from_predefined_query(vectorstore, interest_query)
    career_response = get_response_from_predefined_query(vectorstore, career_query)
    
    if not skill_response:
        skill_response = ""
    if not interest_response:
        interest_response = ""
    if career_response:
        career_formatted = career_response.replace(', ', '\n• ')
    else:
        career_formatted = "정보가 없습니다."
        
    introduce = f"{skill_response}\n\n{interest_response}\n\n• {career_formatted}"
    
    return introduce

def get_response_from_predefined_query(vectorstore, predefined_query):
    # 나머지 코드는 동일하게 유지하되, 'query' 대신 'predefined_query'를 사용
    # k=3으로 설정하여, 가장 유사한 3개의 문서를 가져옴
    docs = vectorstore.similarity_search(query=predefined_query, k=2)
    
    # openai rank lnv process
    llm = ChatOpenAI(temperature=0)
    # llm = OpenAI(model_name = "gpt-3.5-turbo",temperature=0)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=predefined_query)
    
    return response

def get_language_data(vectorstore):
    language_query = "작성자가 학습한 외국어는?(쉼표로 구분, 학습한 '외국어 명' 만을 출력하라, ex. 영어, 일본어)"
    language_response = get_response_from_predefined_query(vectorstore, language_query)
    
    if not language_response or language_response.lower() == 'none':
        return "학습한 외국어 정보가 없습니다."
        
    language_list = language_response.split(', ')
    language_data = []
    
    for lang in language_list:
        test_query = f"{lang}의 평가를 위해 친 시험은?(단답형, '시험 명' 만을 출력하라, 예시) TOEIC, TOEFL, JLPT)"
        test_response = get_response_from_predefined_query(vectorstore, test_query)
        
        score_query = f"{test_response}의 취득 점수를 출력하라?(단답형, 점수만을 출력하라, 예시 : 750)"
        score_response = get_response_from_predefined_query(vectorstore, score_query)
        
        date_query = f"{test_response}의 성적 취득일자를 출력하라?(단답형, 취득일자 만을 출력하라, 예시 : 2017.03)"
        date_response = get_response_from_predefined_query(vectorstore, date_query)
        
        language_data.append({
            "language": lang,
            "test": test_response,
            "date": date_response,
            "score": score_response
        })
    
    return language_data

def get_links(vectorstore):
    link_query = "작성자가 제공하는 노션 및 깃허브 링크는?(쉼표로 구분)"
    link_response = get_response_from_predefined_query(vectorstore, link_query)
    
    if not link_response or link_response.lower() == 'none':
        return "제공되는 노션 및 깃허브 링크가 없습니다."
    
    links_list = link_response.split(', ')
    
    return links_list


#Background images add function
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
#add_bg_from_local('images.jpeg')  

#sidebar contents

# Streamlit의 session state를 초기화하는 함수
def _get_state():
    if 'get' in dir(st.session_state):
        return st.session_state.get('_get_state', {})
    else:
        return {}

# Session State 초기화
def initialize_state():
    if 'name' not in st.session_state:
        st.session_state.name = ""
    if 'introduce' not in st.session_state:
        st.session_state.introduce = ""
    if 'experience' not in st.session_state:
        st.session_state.experience = ""
    if 'education' not in st.session_state:
        st.session_state.education = ""
    if 'skills_list' not in st.session_state:
        st.session_state.skills_list = ""
    if 'awards_and_others' not in st.session_state:
        st.session_state.awards_and_others = ""
    if 'language' not in st.session_state:
        st.session_state.language = ""
    if 'link' not in st.session_state:
        st.session_state.link = ""
    # 여기에 다른 state 초기화 코드 추가

with st.sidebar:
    st.title('🦜️🔗GPT 기반 이력서 양식 변환 서비스')
    st.markdown('''
    ## About APP:

    본 서비스는 BDIA DEV Contest 에 출품한 이력서 양식 변환 서비스 입니다.

    ## How to use:
    
    1. 이력서를 선택 합니다.
    2. 변환 버튼을 누릅니다.(5분 이상 소요될 수 있습니다.)
    3. 변환된 정보를 확인합니다.
    4. "이력서 다운로드"버튼을 누릅니다.
    
    ## Powered by:
    
    - [streamlit](https://streamlit.io/)
    - [Langchain](https://docs.langchain.com/docs/)
    - [OpenAI](https://openai.com/)

    ## Code:

    - [GitHub](https://github.com/ETTE154/Wanted-is-all-you-need.git)
    
    ''')

    add_vertical_space(4)
    

load_dotenv()

def main():
    
    state = _get_state()
    
    st.header("📄이력서 양식 변환 서비스")
    st.subheader("📌Wanted-is-all-you-need")

    #upload a your pdf file
    pdf = st.file_uploader("변환할 이력서를 선택해 주세요.", type='pdf')
        
    if pdf is not None:
        # st.write(pdf.name)
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text+= page.extract_text()

        #langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 2500,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text=text)

        
        #store pdf name
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vectorstore = pickle.load(f)
            #st.write("Already, Embeddings loaded from the your folder (disks)")
        else:
            #embedding (Openai methods) 
            embeddings = OpenAIEmbeddings()

            #Store the chunks part in db (vector)
            vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vectorstore,f)
            
            #st.write("Embedding computation completed")

        if st.button("변환"):
            # 미리 정의된 질문
            with st.spinner("이력서 작성중...(5분이상 소요될 수 있습니다.)"):
            
                get_name = "작성자의 이름은?(단답형, 예시 : 홍길동)"
                get_email = "작성자의 이메일은?(단답형, 예시 : abc123@gmail.com)"
                get_contact = "작성자의 연락처는?(단답형, 예시 : 010-1234-5678)"
                
                # get name
                name = get_response_from_predefined_query(vectorstore, get_name)
                
                # get email
                email = get_response_from_predefined_query(vectorstore, get_email)
                
                # get contact
                contact = get_response_from_predefined_query(vectorstore, get_contact)
                
                # get introduce
                introduce = get_introduce(vectorstore)
                
                # experience = get_response_from_predefined_query(vectorstore, get_experience)
                experience = []
                experience = get_career_info(vectorstore)
                
                # education = get_response_from_predefined_query(vectorstore, get_education)
                education = get_education_info(vectorstore)
                
                # skills_list = get_response_from_predefined_query(vectorstore, get_skills_list)
                skills_list = get_skills_list(vectorstore)
                
                # awards_and_others = get_response_from_predefined_query(vectorstore, get_awards_and_others)
                awards_and_others = get_awards_and_others(vectorstore)
                
                # language = get_response_from_predefined_query(vectorstore, get_language)
                language = get_language_data(vectorstore)
                
                # link = get_response_from_predefined_query(vectorstore, get_link)
                link = get_links(vectorstore)

                st.subheader("이름")
                st.text_area('ex) 홍길동', name)
                
                st.subheader("이메일")
                st.text_area('ex) abc123@gmail.com', email)
                
                st.subheader("연락처")
                st.text_area('ex) 010-1234-5678', contact)                
                                                                    
                st.subheader("자기소개")
                st.text_area('자기소개를 작성해주세요.', introduce)
                
                st.subheader("경력")
                st.text_area('경력을 작성해주세요.', experience)
                
                st.subheader("학력")
                st.text_area('학력을 작성해주세요.', education)
                
                st.subheader("기술")
                st.text_area('기술을 작성해주세요.', skills_list)
                
                st.subheader("수상 및 기타")
                st.text_area('수상 및 기타를 작성해주세요.', awards_and_others)
                
                st.subheader("외국어")
                st.text_area('외국어를 작성해주세요.', language)
                
                st.subheader("링크")
                st.text_area('링크를 작성해주세요.', link)
                
                # filename = (name + "의 이력서.pdf")
                # create_wanted_template_v2(filename = "wanted_template_v2.pdf",
                # applicant_name = "전채욱",
                # email = " a01025648934@gmail.com",
                # contact = " 010-2564-8934",
                # introduce = introduce,
                # experience_data = experience,
                # education_data = education,
                # skills = skills_list,
                # awards_data = awards_and_others,
                # language_data = language_data_sample,
                # links_data=links_data_sample)
                pdf_buffer = create_wanted_template_v2(
                        filename = f"{name}의 이력서.pdf",
                        applicant_name = name,
                        email = email,
                        contact = contact,
                        introduce = introduce,
                        experience_data = experience,
                        education_data = education,
                        skills = skills_list,
                        awards_data = awards_and_others,
                        language_data = language,
                        links_data=link
                    )

                pdf_bytes = pdf_buffer.read()  # BytesIO 객체에서 바이트 데이터를 읽음

                    # 다운로드 버튼 추가
                st.download_button(
                    label="이력서 다운로드",
                    data=pdf_bytes,
                    file_name=f"{name}_이력서.pdf",
                    mime="application/pdf"
)



if __name__=="__main__":
    main()
