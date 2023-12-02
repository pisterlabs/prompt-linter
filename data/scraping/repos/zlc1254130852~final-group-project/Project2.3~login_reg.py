from flask import Blueprint
from flask import request, make_response, render_template, redirect
import random, string, hashlib
from tables import User
from setting import db
from AI_chat import openai_client
from openai import OpenAI
from api_key import OPENAI_API_KEY

login_reg_bp = Blueprint('login_reg', __name__)

@login_reg_bp.route('/login', methods=['GET',"POST"])
def login():
    if request.method=="GET":
        return render_template("login.html")

    req = request.values
    login_name = req['login_name'] if 'login_name' in req else ''
    login_pwd = req['login_pwd'] if 'login_pwd' in req else ''

    user_info = User.query.filter_by(login_name=login_name).first()

    if not user_info:
        return {"msg": "Username has not been registered.", "code": -1}

    if user_info.login_pwd != login_pwd:
        return {"msg": "Please enter the correct password.", "code": -1}

    user_info.login_salt = "".join([random.choice((string.ascii_letters + string.digits)) for i in range(8)])
    # the salt is a random code used to encrypt cookie info, updated every time logged in to prevent the old cookie to be used for "log in"
    db.session.commit()

    m = hashlib.md5()  # encrypt login password
    str = "%s-%s" % (user_info.login_pwd, user_info.login_salt)
    m.update(str.encode("utf-8"))


    response = make_response({"msg": "Login successfully", "code": 200})
    response.set_cookie("Assistant",
                        "%s#%s" % (user_info.id, m.hexdigest()), 60 * 60 * 24 * 7)

    if req['login_name'] not in openai_client:
        openai_client[req['login_name']] = OpenAI(api_key=OPENAI_API_KEY)

    return response

@login_reg_bp.route("/logout")
def logOut():

    cookies = request.cookies
    cookie_name = "Assistant"
    auth_cookie = cookies[cookie_name] if cookie_name in cookies else None

    auth_info = auth_cookie.split("#")
    user_info = User.query.filter_by(id=auth_info[0]).first()
    user_info.login_salt = "".join([ random.choice( (string.ascii_letters + string.digits ) ) for i in range(8) ])
    # update salt info to prevent the old cookie to be used for "log in"
    db.session.commit()

    response = make_response( redirect( "/login" ) )
    response.delete_cookie( "Assistant" )

    return response

@login_reg_bp.route('/reg', methods=['GET',"POST"])
def reg():
    if request.method=="GET":
        return render_template("Reg.html")

    req = request.values
    login_name = req['login_name'] if "login_name" in req else ""
    login_pwd = req['login_pwd'] if "login_pwd" in req else ""

    user_info = User.query.filter_by(login_name=login_name).first()
    if user_info:
        return {"msg": "This user name has already been registered, please use another one.", "code": -1}

    # create new user and store in the database
    model_user = User()
    model_user.login_name = login_name
    model_user.login_pwd = login_pwd
    model_user.login_salt = "".join([random.choice((string.ascii_letters + string.digits)) for i in range(8)])
    db.session.add(model_user)
    db.session.commit()

    return {"msg": "Registered successfully.", "code": 200}