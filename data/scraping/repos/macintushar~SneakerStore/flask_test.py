import openai

from flask import Flask, request, jsonify
from flask_cors import CORS

from db import *
import json

app = Flask(__name__)
CORS(app)

openai.api_key = "GG"

def getSearchJSON(search):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "user", 
        "content": "Read the following sentence and convert it into a JSON object and make sure to highlight keywords like brand, shoe_name, color and size. Input: " + search}])
    
    searchInJSON = completion.choices[0].message["content"]
    searchInJSON = json.loads(searchInJSON)
    return searchInJSON

def solveDoubtGPT(doubt):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "assistant", 
        "content": "Pretend that you are an expert in fashion. " + doubt}])
    
    answerGPT = completion.choices[0].message["content"]
    answerJSON = {"answer":answerGPT}
    print(answerJSON)
    answerJSON = jsonify(answerJSON)
    
    return answerJSON

@app.route("/askGPT")
def askGPT():
    doubt = request.headers['doubt']
    print(doubt)
    
    answer = solveDoubtGPT(doubt)
    print(answer)

    return answer

@app.route("/db_create")
def create_table():
    createTable()
    return("Created")

@app.route("/data")
def getData():
    productData = getShoeData()
    print(productData)

    keys = ("product_id", "product_name", "description", "image", "current_price", "original_price", "discount", "size", "brand", "category", "seller_id", "sku")

    list_of_dicts = [dict(zip(keys, t)) for t in productData]

    json_string = json.dumps(list_of_dicts)
    print(json_string)

    return(json_string) 

@app.route("/addProduct", methods = ['POST'])
def newShoe():
    if request.method == "POST":
        product_name = request.form["product_name"]  
        description = request.form["description"]  
        image = request.form["image"]  
        current_price = request.form["current_price"]
        original_price = request.form["original_price"]
        discount = request.form["discount"]  
        short = request.form["short"]  
        brand = request.form["brand"]
        category = request.form["category"]  
        seller_id = request.form["seller_id"]  
        sku = request.form["sku"]  
    
        return(CreateNewProduct(product_name, description, image, current_price, original_price, discount, brand, category, seller_id, sku, short))

@app.route("/shoe/<shoe>")
def productDetails(shoe):
    print("Name",shoe)
    
    shoeData = GetProduct(shoe)
    shoeData = shoeData[0]
    print(shoeData[0])

    keys = ("product_id", "product_name", "description", "image", "current_price", "original_price", "discount", "size", "brand", "category", "seller_id", "sku")

    dicts = dict(zip(keys, shoeData))

    json_string = json.dumps(dicts)
    print(json_string)

    return([json_string]) 

@app.route("/admin/login", methods = ['POST'])
def AdminLogin():
    if request.method == 'POST':
        email = request.form["email"]  
        password = request.form["password"]
        print(email,password)
        data = LoginAdmin(email, password)
        print(data)
        return (data)
    return ('')
    

if __name__ == '__main__':
   app.run(debug=True, port=5001)