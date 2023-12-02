import bs4
import cv2
import concurrent.futures
from datetime import datetime
import json
import numpy as np
from PIL import Image
import pyautogui
import pytesseract
import requests
import signal
import string
import time
import os
import openai
openai.api_key = os.getenv('OPENAI_API_KEY')
DISC_WEBHK_URL = os.getenv('DISC_WEBHK_URL')
GOOGLE_SEARCH_URL = 'https://www.google.com/search?q='
TIMER_POSITION = (390, 575)
TIMER_COLOR = (125, 249, 175, 255)
QUESTION_REGION = (75, 675, 650, 640)
NUM_ANSWERS = 3
QUESTION_TIME = 10
CHECK_SECONDS = 0.5


def sigint_handler(signum, frame):
    global running
    running = False


class Question:
    def __init__(self, q='', a=[], num=0):
        self._number = num
        self._question = q
        self._answers = a

    def set_question(self, q):
        self._question = q

    def set_answer(self, a):
        self._answers = a

    def set_number(self, n):
        self._number = n

    def get_question(self):
        return self._question

    def get_answer(self, i):
        if i < 0 or i >= len(self._answers):
            raise IndexError('Answer Index does not exist')
        return self._answers[i]

    def get_number(self):
        return self._number

    def get_answers(self):
        return self._answers

    def format_answers(self):
        return '\n'.join(f'{i}. {ans}' for i, ans in enumerate(self._answers, 1))

    def print(self):
        print(f'Question {self._number}: {self._question}')
        print(self.format_answers())

    def get_gpt_prompt(self):
        return f'{self._question} (pick from the {len(self._answers)} options)\n{self.format_answers()}\nAnswer:'

    def get_json(self):
        return json.dumps(self.__dict__)


def detect_color(color, x, y):
    return pyautogui.pixelMatchesColor(x, y, color, 5)


def get_game_img(img_region):
    screenshot = pyautogui.screenshot(region=img_region)
    return screenshot


def process_img(im: Image.Image):
    np_img = np.array(im)
    contrast = 0.8
    brightness = -100
    img = cv2.addWeighted(np.array(np_img), contrast,
                          np.array(np_img), 0, brightness)

    return img


def get_text(im: Image.Image):
    img = process_img(im)
    return pytesseract.image_to_string(img)


def show_text_boxes(im):
    d = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top']
                        [i], d['width'][i], d['height'][i])
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', im)
    cv2.waitKey(0)


def get_question(s, q_num, num_ans):
    split_s = s.split('\n')
    split_s = list(filter(None, split_s))
    question = ' '.join(split_s[:-num_ans])
    question.replace('"', '')
    # assumes the last NUM_ANSWERS lines will always be the answers
    answers = split_s[-num_ans:]
    return Question(question, answers, q_num)


def gen_google_search(q: Question):
    f_answers = '\" OR \"'.join(q.get_answers())
    search_url = f'{GOOGLE_SEARCH_URL}{q.get_question()} "{f_answers}"'
    search_url = search_url.replace(" ", "+")
    return search_url


def make_google_soup(url):
    r = requests.get(url)
    r.raise_for_status()
    return bs4.BeautifulSoup(r.text, 'lxml')


def rem_punc(input_string):
    return input_string.translate(str.maketrans('', '', string.punctuation))


def count_answers(soup: bs4.BeautifulSoup, answers):
    results = dict.fromkeys(answers, 0)
    items = soup.find_all('div')
    for item in items:
        for ans in answers:
            text = item.get_text().lower()
            text_no_punc = rem_punc(text)
            ans_l = ans.lower()
            if ans_l in text or ans_l in text_no_punc:
                results[ans] += 1
    return results


def get_google_results(q: Question):
    search = gen_google_search(q)
    soup = make_google_soup(search)
    return count_answers(soup, q.get_answers())


def get_gpt_ans(prompt):
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[
                                            {"role": "user", "content": prompt}], temperature=0, max_tokens=256, top_p=0.2)
    # print(response)
    return response['choices'][0]['message']['content']


def get_all_answers(q: Question):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(get_google_results, q)
        f2 = executor.submit(get_gpt_ans, q.get_gpt_prompt())
    return f'Google results: {f1.result()}\nGPT3 answer:{f2.result()}'


def create_disc_embed(q: Question, results):
    ans_choices = '\n'.join(
        f'{i}. [{ans}]({(GOOGLE_SEARCH_URL + ans).replace(" ", "+")})' for i, ans in enumerate(q.get_answers(), 1))
    return {'title': f'Question {q.get_number()}: {q.get_question()}', 'url': f'{gen_google_search(q)}', 'description': f'{ans_choices}\n\n**{results}**\n', 'color': 0x000000,
            'fields': [{"name": 'Support this project on Github!',
                        'value': '[star project](https://github.com/peterwzhang/TikTok-Trivia-Helper) or [follow](https://github.com/peterwzhang/TikTok-Trivia-Helper)'
                        }
                       ]}


def post_disc_web_hk(url, q, r):
    data = {'username': 'TikTok Trivia Helper',
            'embeds': [create_disc_embed(q, r)]}
    r = requests.post(url, json=data)


def log_questions(q_list):
    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_name = f'log_{cur_time}.json'
    dir = 'logs'
    os.makedirs(dir, exist_ok=True)
    with open(f'logs/{log_name}', 'w+') as f:
        json.dump([q.get_json() for q in q_list], f, indent=4)
    print(f'\nSaved questions to {dir}/{log_name}')


def run():
    q_list = []
    global running
    running = True
    signal.signal(signal.SIGINT, sigint_handler)
    print('waiting for question...\n')
    while running:
        if detect_color(TIMER_COLOR, *TIMER_POSITION):
            img = get_game_img(QUESTION_REGION)
            screen_text = get_text(img)
            question = get_question(screen_text, len(q_list) + 1, NUM_ANSWERS)
            question.print()
            results = ''
            if openai.api_key:
                results = get_all_answers(question)
            else:
                results = f'Google results: {get_google_results(question)}'
            print(results)
            q_list.append(question)
            if DISC_WEBHK_URL:
                post_disc_web_hk(
                    DISC_WEBHK_URL, question, results)
            print('\nwaiting for question...\n')
            time.sleep(QUESTION_TIME)
        else:
            time.sleep(CHECK_SECONDS)
    log_questions(q_list)


def main():
    run()


if __name__ == "__main__":
    main()
