from flask import Flask, render_template, request, redirect, url_for, session
import openai
from gpt import gpt_interior, gpt_feature, gpt_number,gpt_translate,gpt_translate_ja
from clip_code import *
from other_code import *
app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == "GET":
        return render_template('index_ja.html')
    else:
        if "start" in request.form:
            return redirect(url_for("search"))
        else:
            global interior, features, number,device
            input_number = request.form["number"]
            device = request.form["device"]
            input_text = request.form["keyword"]
            input_text = gpt_translate(input_text)
            interior = gpt_interior(input_text,input_number)
            features = gpt_feature(input_text,interior)
            number = gpt_number(input_text,interior)
            ja_interior = gpt_translate_ja(",".join(interior))
            
            ja_features = gpt_translate_ja(",".join(features))
            return render_template('index_ja.html',interior=ja_interior,features=ja_features,number=number)

@app.route("/search",methods=['GET',"POST"])
def search():
    global interior, features, number,device
    if request.method == "GET":
        return render_template('search_ja.html',interior=interior,features=features,number=number)
    else:
        if "object_uid" in request.form:
            uid = request.form["object_uid"]
            return redirect(url_for("watch", uid=uid))
        else:
            names, urls, uids = squeeze(interior)
            selected_name, selected_uid = get_uid(interior, features, names,urls, uids,device)
            install_obj(selected_uid)
            names = save_img(selected_uid, "./data.json")
            return render_template('search_ja.html',interior=interior,features=features,number=number,selected_name=selected_name,selected_uid=selected_uid)
    
@app.route("/watch", methods=['GET', 'POST'])
def watch():
    if request.method == "GET":
        uid = request.args.get("uid", None)
        if uid:
            return render_template('watch.html', uid=uid)
        else:
            return "No uid provided", 400

    
if __name__ == "__main__":
    app.run(debug=True)