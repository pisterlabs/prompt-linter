from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import timedelta
import urllib
import requests
import openai
import hashlib
import json
from app import *
from sqlalchemy import or_, and_, not_

OPENAI_API_KEY = ""
openai.organization = ""
openai.api_key = OPENAI_API_KEY

# navigate to login page


@app.route('/', methods=['GET', 'POST'])
@app.route('/toLogin', methods=['GET', 'POST'])
def toLogin():
    return render_template('login.html')

# navigate to register page


@app.route('/toRegister', methods=['GET', 'POST'])
def toRegister():
    return render_template('register.html')

# navigate to main page


@app.route('/toChatBot/<token>', methods=['GET', 'POST'])
def toChatBot(token):
    if token_validate(token):
        return render_template('chatbot.html', token=token)


@app.route('/returnMessage/<token>', methods=['GET', 'POST'])
def returnMessage(token):
    if token_validate(token):
        send_message = request.values.get("send_message")
        print("user message：" + send_message)
        query = Statement(content=send_message, title="User",
                          conversation_id=request.values.get("con_id"))
        db.session.add(query)
        # openAI version
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                  messages=[{"role": "user", "content": send_message}])
        message = completion.choices[0].message.content
        print("Chatgpt: ", message)
        reply = Statement(content=message, title="Bot",
                          conversation_id=request.values.get("con_id"))
        db.session.add(reply)
        db.session.commit()
        return message
    else:
        return "Token Invalid!"


@app.route('/newConversation/<token>', methods=['GET', 'POST'])
# auxiliary function, to start a new conversation
def newConversation(token):
    if token_validate(token):
        new_conversation = Conversation(title=request.values.get(
            "title"), user_id=request.values.get("user_id"))
        db.session.add(new_conversation)
        db.session.flush()
        db.session.commit()
        data = {
            "id": new_conversation.id,
            "startTime": new_conversation.startTime.strftime('%Y-%m-%d %H:%M:%S')
        }
        return {"status": 200, "data": json.dumps(data)}
    else:
        print("Token invalid")
        return {"status": 404}


@app.route('/updateConversationTitle/<token>', methods=['POST'])
def updateConversationTitle(token):
    if token_validate(token):
        con_id = request.values.get("con_id")
        new_title = request.values.get("new_title")

        if not all([con_id, new_title]):
            return jsonify(status=400, message="Missing con_id or new_title")

        conversation = Conversation.query.filter_by(id=con_id).first()
        if conversation:
            conversation.title = new_title
            db.session.commit()
            return jsonify(status=200, message="Title updated successfully")
        else:
            return jsonify(status=404, message="Conversation not found")
    else:
        return jsonify(status=403, message="Invalid token")


@app.route('/refreshConversations/<token>', methods=['GET', 'POST'])
# auxiliary function, to start a new conversation
def refreshConversations(token):
    if token_validate(token):
        user_id = request.values.get("user_id")
        keyword = request.values.get("keyword")
        if keyword:
            conversations = Conversation.query.filter(and_(Conversation.user_id == user_id,
                                                           Conversation.title.contains(keyword))).all()
        else:
            conversations = Conversation.query.filter_by(user_id=user_id).all()
        data = []
        for con in conversations:
            data.append({
                'id': con.id,
                'title': con.title,
                'startTime': con.startTime.strftime('%Y-%m-%d %H:%M:%S')
            })
        return {"status": 200, "data": json.dumps(data)}


@app.route('/fetchConversation/<token>', methods=['GET', 'POST'])
# auxiliary function, to start a new conversation
def fetchConversation(token):
    if token_validate(token):
        con_id = request.values.get("con_id")
        statements = Statement.query.filter_by(conversation_id=con_id).all()
        data = []
        for item in statements:
            data.append({
                'id': item.id,
                'title': item.title,
                'content': item.content,
                'timeStamp': item.timeStamp.strftime('%Y-%m-%d %H:%M:%S')
            })
        return {"status": 200, "data": json.dumps(data)}


# Automatically generate a new conversation when logging in
@app.route('/login', methods=['GET', 'POST'])
def logIn():
    md5 = hashlib.md5()  # Create md5 encryption object
    md5.update(request.values.get("password").encode(
        'utf-8'))  # String to be encrypted
    pwd_md5 = md5.hexdigest()  # Encrypted String
    user = User.query.filter_by(
        name=request.values.get("account"), pwd=pwd_md5).first()
    if user:
        # create token
        original_token = (
            user.name + str(datetime.now().timestamp())).encode('utf-8')
        md5 = hashlib.md5()  # Create new md5 encryption object
        md5.update(original_token)  # String to be encrypted
        token = md5.hexdigest()  # Encrypted String
        new_token = Token(content=token, user_id=user.id)
        db.session.add(new_token)
        db.session.commit()
        data = {
            "url": url_for("toChatBot", token=token),
            "token": token,
            "user": {
                "name": user.name,
                "id": user.id
            }
        }
        return {"status": 200, "data": json.dumps(data)}
    else:
        return {"status": 404}


@app.route('/autoLogin', methods=['GET', 'POST'])
def autoLogIn():
    token = request.values.get("token")
    if token_validate(token):
        TOKEN = Token.query.filter_by(content=token).first()
        data = {
            "url": url_for("toChatBot", token=token),
            "user": {
                "id": TOKEN.User.id,
                "name": TOKEN.User.name
            }
        }
        return {"status": 200, "data": json.dumps(data)}
    else:
        return {"status": 404}


@app.route('/register', methods=['GET', 'POST'])
def register():
    md5 = hashlib.md5()  # Create md5 encryption object
    md5.update(request.values.get("password").encode(
        'utf-8'))  # String to be encrypted
    pwd_md5 = md5.hexdigest()  # Encrypted String
    new_user = User(name=request.values.get("account"), pwd=pwd_md5)
    db.session.add(new_user)
    db.session.commit()
    return url_for("toLogin")

# auxiliary function, to validate token


def token_validate(token_md5):
    token = Token.query.filter_by(content=token_md5).first()
    if not token:
        return False
    Duration = 24 * 60 * 60  # Token is valid for 24 hours
    TS_Now = datetime.now()
    if TS_Now.timestamp() - token.timeStamp.timestamp() >= Duration:
        # Token is invalid
        db.session.delete(token)
        db.session.commit()
        return False
    else:
        # Update the timestamp of the token
        token.timeStamp = TS_Now
        db.session.commit()
        return True


@app.route('/logout', methods=['POST'])
def logout():
    token = request.values.get("token")
    if token:
        TOKEN = Token.query.filter_by(content=token).first()
        if TOKEN:
            db.session.delete(TOKEN)
            db.session.commit()
            return {"status": 200, "message": "Logged out successfully"}
        else:
            return {"status": 404, "message": "Token not found"}
    else:
        return {"status": 400, "message": "No token provided"}


if __name__ == '__main__':
    app.run(debug=True)
