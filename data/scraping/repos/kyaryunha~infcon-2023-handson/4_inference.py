import openai
import requests
from bs4 import BeautifulSoup

from api_key import api_key
openai.api_key = api_key


site_url = 'https://www.inflearn.com'


def get_data_by_path(course_path):
    course_url = f'{site_url}{course_path}'
    response = requests.get(course_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    course_title = soup.find('h1', class_='cd-header__title').text.strip()
    instructor_name = soup.find('a', class_='cd-header__instructors--main').text.strip()
    course_tags = soup.find_all('a', class_='cd-header__tag')
    course_tag_list = []
    for course_tag in course_tags:
        course_tag_list.append(course_tag.text.strip())

    course_description = soup.find('section', 'cd-body')
    return {
        'prompt': f"강의 설명': \n{course_title} ({instructor_name}) - {' '.join(course_description.stripped_strings)[:800]}. \n 위의 강의 설명에 해당하는 태그는 다음과 같습니다.\n태그: \n",
        'completion': ', '.join(course_tag_list),
    }


res = get_data_by_path('/course/chatgpt-다국어-번역기-웹개발')
prompt = res['prompt']
answer = res['completion']

response = openai.Completion.create(
    engine="davinci:ft-hyun-2023-08-10-03-20-56",
    prompt=prompt,
    max_tokens=50,
    temperature=0,
)

print("------------------------")
print("api 응답 결과")
print(response.choices[0]["text"])
print("------------------------")
print("정답")
print(answer)
