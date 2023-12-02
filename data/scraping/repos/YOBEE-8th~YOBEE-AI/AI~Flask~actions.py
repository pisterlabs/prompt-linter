import requests
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
import openai
from api_key import OPENAI
import logging

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.logger.setLevel(logging.INFO)
cors = CORS(app, resources={
    r"/*": {
        "origins": "*"
        }
    })

# 클라이언트 요청 -> RASA (요청 -> 응답) -> 정보 가공 -> 클라이언트 응답

@app.route('/api/v1/voice', methods=['POST'])
def rasa_request():
# 클라이언트 요청
    recipeId = request.json['recipeId']
    message = request.json['message'] # 클라이언트로부터 메시지를 받아옴
    token = request.headers.get('Authorization')
    if token is None:
        return jsonify({'error': 'No token provided.'}), 401
    
# Rasa 서버에 요청을 보내고 응답을 받음
    rasa_url = "http://13.125.182.198:5005/webhooks/rest/webhook"
    # rasa_url = "http://localhost:5005/webhooks/rest/webhook"
    rasa_payload = {"message": message}
    rasa_response = requests.post(rasa_url, json=rasa_payload).json()
    # print(rasa_response)
# 정보 가공
    result_type = ''
    result_text = ''
    result_message = ''
    # type: 페이지 이동 (result_type = 0)
    if 'page' in rasa_response[0]['text']:
        result_type = '0'
        result_message = "페이지 이동 명령"
        if rasa_response[0]['text'] == "first_page":
            result_text = '0'
            result_message += "( << )"
        elif rasa_response[0]['text'] == "previous_page":
            result_text = '1'
            result_message += "( < )"
        elif rasa_response[0]['text'] == "next_page":
            result_text = '2'
            result_message += "( > )"
        elif rasa_response[0]['text'] == "last_page":
            result_text = '3'
            result_message += "( >> )"
        else:
            result_text = '4' # current_page
            result_message += "( ● )"

    # type: 재료 질문
    elif '재료:' in rasa_response[0]['text']:
        ingredient = rasa_response[0]['text'].replace("재료:","")
        result_type = '1'
        result_message = "재료질문 답변"

        # Spring 서버 API 주소
        spring_url = "http://13.125.182.198:8090/api/v1/AI/recipe/"+ str(recipeId) +"/ingredient"
        # HTTP 요청 헤더 설정
        headers = {"Authorization": token}
        # 요청 보내기 
        spring_response = requests.get(spring_url, headers=headers)
        # 응답 데이터 가져오기
        spring_response = spring_response.json()
        # print(spring_response)
        ingredients = spring_response['data']

        # print("ingredients:", ingredients)
        for ingredient_name, ingredient_weight in ingredients.items():
            
            if ingredient == ingredient_name:
                ingredient_weight = ingredient_weight.replace('g', '그램')
                ingredient_weight = ingredient_weight.replace('t', '작은술')
                ingredient_weight = ingredient_weight.replace('T', '큰술')
                ingredient_weight = ingredient_weight.replace('L', '리터')
                ingredient_weight = ingredient_weight.replace('ml', '밀리리터')

                result_text = ingredient_name + ' ' + ingredient_weight +' 넣어주세요'
                break
        else:
            result_text = ingredient + ' 들어가지 않아요.'

    else:
        result_type = '1'
        result_message = "ChatGPT 답변"

        spring_url = "http://13.125.182.198:8090/api/v1/recipe/unresolved"
        headers = {"Authorization": token, "Content-Type": "application/json"}
        data = {"recipeId": recipeId, "content": message}
        spring_response = requests.post(spring_url, json=data, headers=headers)
        spring_response = spring_response.json()
        # print(spring_response)

        # 발급받은 API 키 설정 (호스트의 디렉토리를 볼륨하는 방법)
        OPENAI_API_KEY = OPENAI.OPENAI_API_KEY

        openai.api_key = OPENAI_API_KEY
    

        engine = "text-davinci-003"
        model = "text-davinci-003"
        prompt = message + " 한국어로 요약해서 말해줘. "
        temperature = 0.5
        max_tokens = 1000

        completions = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            # model=model
        )

        result_text = completions.choices[0].text.replace('\n','')


# 클라이언트 응답
    response = {
        'status': 200,
        'message': result_message,
        'data':{
            'type': result_type,
            'text': result_text
        }
    }

    response = make_response(jsonify(response))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    app.logger.info(f'')
    app.logger.info(f'user_input: {message}')
    app.logger.info(f'result_text: {result_text}')
    app.logger.info(f'result_message: {result_message}')
    app.logger.info(f'')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006)
    
