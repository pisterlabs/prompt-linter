from bs4 import BeautifulSoup
import pandas as pd
import requests
import regex as re
import cohere
import os
import json
from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
app = Flask(__name__)

co = cohere.Client(os.environ['CO_API_KEY'])

email = re.compile('[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+[.][A-Za-z.]{2,}')
name = re.compile('[A-Z][a-zA-Z]')

@app.route("/api/v1/emails", methods=["POST"])
def get_emails():
    data = json.loads(request.data)
    url_format = re.compile("((http|https)://)(www.)?[a-zA-Z0-9@:%._\\+~#?&//=]{2,256}\\.[a-z]{2,6}\\b([-a-zA-Z0-9@:%._\\+~#?&//=]*)")
    if not "url" in data:
      return "Invalid json data format"
    url = data['url']
    if not url_format.match(url):
      return "Invalid URL"
    req = requests.get(url)

    content = req.text

    soup = BeautifulSoup(content, features="html.parser")

    emails = email.findall(str(soup.find('main')))
    emails = list(dict.fromkeys(emails))

    return jsonify(emails)

@app.route("/api/v1/csv", methods=["POST"])
def generate_csv():
    data = json.loads(request.data) 
    names = data['names']
    emails = data['emails']
    if (not type(names) is list):
       return "Invalid names"
    if (not type(emails) is list):
       return "Invalid emails"
    size = len(emails)
    status = ["Not Applied" for i in range(size)]
    contacted = ["No" for i in range(size)]
    recruiters = { "names" : names, "emails": emails, "status": status, "contacted": contacted }
    
    df = pd.DataFrame(recruiters, columns=["names", "emails", "status", "contacted"]).set_index('names')
    if not os.path.isfile('recruiters.csv'):
      df.to_csv('recruiters.csv', header=['emails', 'status', 'contacted'])
    else: # else it exists so append without writing the header
      pass
      # df.to_csv('recruiters.csv', mode='a', header=False)

@app.route("/api/v1/csv") 
def getPlotCSV():
    if not os.path.isfile('recruiters.csv'):
        return None
    df = pd.read_csv('recruiters.csv')
    csv = df.to_csv()[1:]
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=myplot.csv"})


@app.route("/api/v1/research", methods=["POST"])
@cross_origin()
def get_research_template():
    data = json.loads(request.data)
    student_name = data['student_name']
    student_field = data['student_field']
    student_experience = data['student_experience']
    student_uni = data['student_uni']
    prof_name = data['prof_name']
    prompt = f"Write a cold outreach email to a professor named {prof_name} from a student named {student_name}, who is a {student_experience} at {student_uni}, asking if {prof_name} is interested in hiring {student_name} as a research assistant regarding {student_field}. Do not generate emails or phone numbers. Only ask if they are open to hiring people."
    response = co.generate(
    model='command-xlarge-nightly',
    prompt=prompt,
    max_tokens=300,
    temperature=0.5,
    k=0,
    stop_sequences=[],
    return_likelihoods='NONE')
    return jsonify({ "text": response.generations[0]})

@app.route("/api/v1/internship", methods=["POST"])
@cross_origin()
def get_internship_template():
    data = json.loads(request.data)
    student_name = data['student_name']
    student_field = data['student_field']
    student_experience = data['student_experience']
    student_uni = data['student_uni']
    company_name = data['company_name']
    prompt = f"Write an outreach cold email written by a student named {student_name} interested in working in the field of {student_field}. The person is a {student_experience} at {student_uni}, and is asking about an internship at a company called {company_name}. Do not generate emails or phone numbers. Only ask if they are open to hiring people."
    response = co.generate(
      model='command-xlarge-nightly',
      prompt=prompt,
      max_tokens=300,
      temperature=0.5,
      k=0,
      stop_sequences=[],
      return_likelihoods='NONE')
    return jsonify({ "text": response.generations[0] })

def get_names(url):
    req = requests.get(url)

    content = req.text

    soup = BeautifulSoup(content, features="html.parser")

    names = []

    for div in soup.findAll('td'):
      try:
        names.append(div.find('a').contents[0])
      except:
        continue

    names = [i for i in names if not email.match(i)]

    return jsonify(names)
    
app.run(debug=False)

