import os
import time

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.llms import OpenAI
from langchain.docstore import Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

from langchain.agents.react.base import DocstoreExplorer

"""
종류가 다양해서 뭐가 에이전트라는 것은 따로 없음.

앞에서 체인은 미리 정해진 기능을 수행하는 것.
에이전트는 사용자의 요청에 따라 제공된 도구를 수행하는 것.

에이전트: 
[입력 → 추론(Reason) → 행동(Action) → 관찰 → 완료]

주로, 추론과 행동을 반복해서 원하는 결과가 나올 때까지 번갈아가며 반복하는 것을 구글이 제시한 'ReAct' 방식을 기반으로 합니다.

ex) 
파이썬 프로그램 짜줘 → 생성 → 인터프리터 실행(도구) → 오류 → 다시 생성 → 인터프리터 실행(도구) → 정상작동(혹은 정답과 비교) → 완료
"""

def main():
    docstore = DocstoreExplorer(Wikipedia())

    tools = [
        Tool(
            name="Search",
            func=docstore.search,
            description="useful for when you need to ask with search",
        ),
        Tool(
            name="Lookup",
            func=docstore.lookup,
            description="useful for when you need to ask with lookup",
        ),
    ]

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True, handle_parsing_errors=True)

    question = "저자 데이비드 채노프는 어느 대통령 밑에서 주영 대사를 역임한 미 해군 제독과 협업한 적이 있나요?"
    question = "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?"
    react.run(question)

    """
    생각: 데이비드 채노프와 미 해군 제독을 찾고, 영국 주재 대사를 찾은 다음, 그들이 근무했던 대통령을 찾아야 합니다.
    실행: 검색[데이비드 채노프]
    관찰: 데이비드 채노프는 논픽션으로 유명한 작가입니다. 그의 작품은 주로 해당 작품의 주인공과 공동 작업을 하는 경우가 많습니다. 그의 공동 작업자로는 어거스터스 A. 화이트, 조이셀린 엘더스, 도안 반 토아이, 윌리엄 J. 크로우, 아리엘 샤론, 케네스 굿, 펠릭스 잔드먼 등이 있습니다. 또한 워싱턴 포스트, 뉴 리퍼블릭, 뉴욕 타임즈 매거진에 문학사, 교육, 외국 등 다양한 주제에 대한 글을 기고했습니다. 12권 이상의 책을 출간했습니다.
    
    생각: 데이비드 채노프는 주영국 대사를 역임한 미 해군 제독과 협력한 적이 있습니다. 미 해군 제독을 검색하여 영국 주재 대사를 찾아야겠어요.
    실행: 검색[미 해군 제독]
    관찰 완료: 해군 제독은 1944년 함대 제독이 만들어지기 전까지 미국 해군에서 가장 높은 계급이었습니다. 이 계급은 적어도 5성 제독과 동등한 것으로 간주되며, 조지 듀이 제독이 이 계급에 임명된 유일한 장교입니다.
    
    생각: 이 단락에는 주영국 대사에 대한 언급이 없습니다. 영국 주재 미 해군 제독 대사를 검색해야겠어요.   
    액션: 검색: [영국 주재 미 해군 제독 대사] 찾기
    관찰: 영국 주재 미 해군 제독 대사]를 찾을 수 없습니다. 유사어: ['최초의 해군 잭', '미국 해군 4성 제독 명단', '레이몬드 A. 스프루언스', '미국 해군 장관', '영국-미국 관계', '윌리엄 J. 크로우', '케네스 브라이트웨이트', '공로 훈장(미 육군)', '미국 해군 유럽 및 아프리카 부대', '미국 혁명의 아들 회원 명단']
    
    생각: 윌리엄 J. 크로우를 검색하여 주영국 대사를 찾을 수 있겠다.
    실행: 검색[윌리엄 J. 크로우]
    관찰: 윌리엄 제임스 크로우 주니어(1925년 1월 2일 - 2007년 10월 18일)는 미국 해군 제독이자 외교관으로 로널드 레이건 대통령과 조지 H. W. 부시 대통령 밑에서 제11대 합참의장을 역임했고, 빌 클린턴 대통령 밑에서 영국 대사와 정보감독위원회 의장을 지냈어요.
    
    생각: 윌리엄 J. 크로우는 빌 클린턴 대통령 시절 주영국 대사였으므로 정답은 빌 클린턴입니다.
    실천하기: 완료[빌 클린턴]
    
    > 체인 완료.

    """
    pass

if __name__=="__main__":
    main()
    pass