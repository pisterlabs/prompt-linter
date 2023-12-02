from flask import Blueprint
from flask import render_template
from flask import request
from AI_chat import openai_client
from d_id_talk import generate_video
from login_check import check_login

video_chat_bp = Blueprint('video_chat', __name__)

@video_chat_bp.route('/video_chat',methods=['GET'])
def video_chat():
    user_info = check_login()  # check which user is logged in.

    if user_info:  # if there is a logged-in user
        return render_template("video.html", current_user=user_info.login_name)
    else:
        return render_template("video.html")

@video_chat_bp.route('/video',methods=['POST'])
def video():
    req = request.values
    current_user = req["current_user"]
    question = req["question"]
    print(question)
    response = openai_client[current_user].chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": question}],
    )

    result=generate_video(response.choices[0].message.content)
    print(result)
    return {"result": result}