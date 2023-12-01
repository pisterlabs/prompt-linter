
import openai
import io
import base64
from PIL import Image
import requests
import cv2
import numpy as np
import random
from rembg import remove
import deepl

def deepl_trans(message, tar_lan):
    auth_key = "be790c4a-9c58-a5fc-4f6a-bea242f8f16f:fx"
    translator = deepl.Translator(auth_key)
    result = translator.translate_text(message, target_lang=tar_lan)
    return result.text

def make_noise_disk(H, W, C, F):
    noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F: F + H, F: F + W]
    noise -= np.min(noise)
    noise /= np.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise

def shuffle(img, h=None, w=None, f=None):
    H, W, C = img.shape
    if h is None:
        h = H
    if w is None:
        w = W
    if f is None:
        f = 256
    x = make_noise_disk(h, w, 1, f) * float(W - 1)
    y = make_noise_disk(h, w, 1, f) * float(H - 1)
    flow = np.concatenate([x, y], axis=2).astype(np.float32)
    return cv2.remap(img, flow, None, cv2.INTER_LINEAR)

def stringToRGB(base64_string):
    imgdata = base64.b64decode(base64_string)
    dataBytesIO = io.BytesIO(imgdata)
    image = Image.open(dataBytesIO)
    return image

def generate_lineart(keyword):
    f = open("/home/ubuntu/config.txt", 'r')
    f.readline()
    endpoint = f.readline().strip()
    
    payload = {
        "width": 512,
        "height": 512,
        "prompt": f"one {keyword}, lineart, simple background, <lora:animeLineartMangaLike_v30MangaLike:1>",
        "negative_prompt" : "nsfw",
        "sd_model_checkpoint": "anything-v4.5.safetensors [1d1e459f9f]",
        "steps": 20,
    }

    response = requests.post(url=f'{endpoint}/sdapi/v1/txt2img', json=payload)
    r = response.json()
    image = stringToRGB(r['images'][0])
    img_resize_lanczos = image.resize((256, 256), Image.LANCZOS)

    imgByteArr = io.BytesIO()
    img_resize_lanczos.save(imgByteArr, format=img_resize_lanczos.format)
    imgByteArr = imgByteArr.getvalue()
    image_file_object = imgByteArr

    r = requests.post('https://clipdrop-api.co/remove-background/v1',
                      files={
                          'image_file': ('portrait.jpg', image_file_object, 'image/jpeg'),
                      },
                      headers={
                          'x-api-key': '1b8fb6c83ab1a0c1ea03492fd6cc940606aba7ee728fa31a560eb4a36eb850090a8fc09a36b9d90e1d185ad828db0cc2'}
                      )
    base64_string = 0
    if (r.ok):
        image_n = Image.open(io.BytesIO(r.content))
        base64_string = base64.b64encode(image_n.read())
    else:
        r.raise_for_status()
        return 0
    #image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_LANCZOS4)
    #image = remove(image)
    #image = cv2.imencode('.png', image)
    #base64_string = base64.b64encode(image[1]).decode()
    return base64_string


def questionMaking(age, num, qa_dict):
    type = ""
    world_view = ""
    
    for i in range(num):
        type += f"{i+1}번: ({i+1}번 질문) '+' '(답변 예시)','(답변 예시)','(답변 예시)' \n"
    for key, value in qa_dict.items():
        world_view += f"- {key} {value} \n"
    
    prompt = f'''
당신은 대한민국의 {age}살 동화 작가와 대화하고 있습니다. 
보기의 세계관을 참고하여, 동화 내용에 대한 적절한 질문을 하시오.

질문은 무조건 {num}개만 할 수 있고, 한번에 하나의 질문만 해야합니다.
질문은 동화를 스토리를 위해서 창의성이 있어야 합니다.
질문은 보기와 관련없어도 됩니다.
작가는 {age}살이므로, 질문은 나이 수준을 고려해야 합니다.
반드시 정확한 캐릭터 이름을 사용하시오.

다음의 형식을 따라 질문하세요.
질문 번호가 커질수록, 스토리의 후반부 질문을 하시요.
{type}
보기(세계관):
{world_view}
'''
    print(prompt)
    return prompt
    # 이걸 gpt api로 보내고 받은 결과물을 반환해주면 될듯


def storyMaking_prompt(num, qa_dict):

    type = ""
    world_view = ""
    
    for i in range(num):
        type += f"문단{i+1}: (내용) \n"
    for key, value in qa_dict.items():
        world_view += f"- {key} {value} \n"
    
    prompt = f'''
당신은 대한민국의 최고의 동화 작가입니다.
아래의 세계관을 참고하여서 5살이 읽을 동화책을 작성하시오.

스토리가 매끄럽게 이어지지 않는다면, 세계관을 추가해도 됩니다.
스토리는 구체적이고 창의적으로 써야합니다.
답변은 반드시 동화의 스토리만 다루시오.
스토리는 자연스럽게 이어져야 합니다.
반드시 한 문단은 2문장입니다.

세계관:
{world_view}
반드시 다음의 형식에 맞게 답변하시오.
{type}

'''
    print(prompt)
    return prompt


def characterMaking(world_view):
    
    prompt = f'''
당신은 이 동화에서 캐릭터를 추출해야 합니다.
스토리를 참고하여, 캐릭터를 일반명사로 쓰시오. 고유명사는 빼고 쓰시오.
캐릭터의 이름이 아닌, 종(species) 이름을 쓰시오.
인간인 경우는 성별을 포함하여 쓰시오.
단수형으로 쓰시오.
캐릭터는 최대 5명까지만 쓰시오.

스토리
{world_view}

다음의 형식을 따라 영어로 답변하세요.
등장인물1: "(영어 일반명사)"
등장인물2: "(영어 일반명사)"
등장인물3: "(영어 일반명사)"
등장인물4: "(영어 일반명사)"
등장인물5: "(영어 일반명사)"

'''
    print(prompt)
    return prompt
    # 이걸 gpt api로 보내고 받은 결과물을 반환해주면 될듯


def storyToBackground_prompt(stories):

    story = ""
    type = ""
    for i in stories:
        story += i+'\n'
    for i in range(len(stories)):
        type += f"문단{i+1}:'(배경 설명)'\n"

    
    
    prompt = f'''
보기의 스토리를 참고하여, 각 문단마다 장소 정보를 추출하시오.
등장 캐릭터을 있는 장소에 대해서 영어로 5단어 이내로 서술하시오.
생명체와 관련된 영어 단어는 제외하시오.

[스토리]
{story}
아래 형식에 맞게 답변하시오.
{type}
'''
    print(prompt)
    return prompt


def openai_api(prompt, model_name):
    f = open("/home/ubuntu/config.txt", 'r')
    api_key = f.readline().strip()

    openai.api_key = api_key #api key 입력해야 함

    response = openai.ChatCompletion.create(
    model=model_name,
    messages=[
        {"role": "system", "content": prompt}
      ]
    )
    output_text = response["choices"][0]["message"]["content"]
    return output_text

def story_generate(dto_json, len_sentence):
    qa_dict = {}

    for i in dto_json:
        question = i['question']['question']
        options = i['option']
        qa_dict[question] = options

    prompt = storyMaking_prompt(len_sentence, qa_dict)
    response = openai_api(prompt, "gpt-4-1106-preview")
    response = response.split("\n")
    refined_response = [i for i in response if i != '']
    #예외처리
    if len(refined_response) == len_sentence * 2:
        temp = []
        for i in range(0,len(refined_response), 2):
            temp.append(refined_response[i] + refined_response[i+1])
        refined_response = temp

    return refined_response

def background_generate(refined_response):
    f = open("/home/ubuntu/config.txt", 'r')
    f.readline()
    endpoint = f.readline().strip()

    url = endpoint # webui endpoint

    prompt = storyToBackground_prompt(refined_response)
    bg_response = openai_api(prompt, "gpt-4-1106-preview") #"gpt-3.5-turbo"
    bg_response = bg_response.split("\n")

    #걍 전처리
    bg_prompt = []
    for i in bg_response:
        bg_prompt.append(i.split(':')[-1].replace("'","").strip())

    #sd 요청 payload
    control_img = cv2.imread('./asset/crayon.png')
    control_img = shuffle(control_img)
    control_img = cv2.imencode('.png', control_img)
    base64_string = base64.b64encode(control_img[1]).decode()
    
    payload = {
        "width": 1024,
        "height": 512,
        "negative_prompt" : "human, animal",
        "sd_model_checkpoint": "anything-v4.5.safetensors [1d1e459f9f]",
        "steps": 20,
        "alwayson_scripts" : {
                "controlnet": {
                "args": [
                        {
                            "input_image": base64_string,
                            "model"  : "control_v11e_sd15_shuffle [526bfdae]",
                            "weight" : 0.45,
                        }
                    ]
                }
            }
    }

    total_image = []
    
    #sd 생성
    for prompt in bg_prompt:
        payload["prompt"] = f"{prompt}, background, crayon style"
        response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
        r = response.json()
        total_image.append(r['images'][0])
    
    return total_image


def making_init_question():
    first = "주인공의 이름은?"
    second = "주인공의 성별은?"
    third = [
        "주인공이 좋아하는 노래는?",
        "주인공이 좋아하는 색깔은?",
        "주인공이 좋아하는 놀이는?",
        "주인공이 좋아하는 계절은?",
        "주인공이 좋아하는 동물은?",
        "주인공의 가장 특별한 추억은?"
    ]
    forth = [
        "주인공의 가장 친한 친구의 이름은?,주인공이 처음으로 만나는 친구의 성격은?",
        "주인공의 가장 친한 친구의 이름은?,주인공의 친구가 가진 특별한 능력은?",
        "주인공에게 가장 큰 영감을 주는 가족은?",
        "주인공이 가족과 함께하는 특별한 기념일은?"
    ]
    fifth = [
        "동화 속 왕국의 이름은?",
        "동화 속에 나오는 특별한 장소는?",
        "동화 속에 사람들은 먹는 음식은?",
        "동화 속에 존재하는 특별한 날씨현상은?",
        "동화 속에 존재하는 특별한 축제는?"
    ]
    sixth = [
        "주인공이 소유한 특별한 물건은?",
        "주인공이 마주치는 어려움은?",
        "주인공이 가장 감명 깊게 배우는 교훈은?",
        "주인공이 꿈꾸는 모험의 목적지는?",
        "이야기에 등장하는 악당의 이름은?"
    ]


    total = [first, second, random.choice(third)] + random.choice(forth).split(',') + [
        random.choice(fifth), random.choice(sixth)]
    
    r_format = {
        "questionData": []
    }
    for i in total:
        one_qa = {
            "question": i
        }
        r_format["questionData"].append(one_qa)

    return r_format