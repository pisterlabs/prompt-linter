# langchain
# pandas
# subprocess
# Flask
# flask_restful
# langchain
# openai
# matplotlib
# mpld3


from flask import Flask, render_template, request,url_for, jsonify, redirect
from flask_restful import Api, Resource
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import pandas as pd
import subprocess
import json

app = Flask("__name__")
api = Api(app)
pd.set_option('display.max_columns', None)

chat_messages=[]
visualizationMode : bool = False
summarize_prompt = """
Summarize the following set of information

{information}

"""

csv_visualize_template = """
    Assume all libraries exist

    Given below is information about a csv file with an example on how the values are stored.

    The name of the file is {filename}

    {csv_example}

    Answer the query given below by generating python code that gives the correct result for the query.

    {query} 
"""


chain : LLMChain = None

def init_chain(template) -> LLMChain:
    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0, client=None, verbose=True)
    globals()["chain"] = LLMChain(llm=llm, prompt=prompt, verbose=True)
    return chain

init_chain(csv_visualize_template)

count = 1
def parse_and_run_output(output : str) -> str:
    # For Demo Purposes, there's only one session and only one graph per query
    print(output)
    save_filename = fr"static/assets/session/graph{globals()['count']}"
    try:
        value = output.split("```")[1][7:] # Python Code extracted
    except Exception as e:
        value = output
        print(e)
        
    new_val = value.replace("plt.show()", 
                            f"""
fig = plt.gcf()
mpld3.save_html(fig, r"{save_filename}.html")
metadata=mpld3.fig_to_dict(fig)""")

    metadata = """
meta_dict = {'width': metadata['width'], 'height':metadata['height']}
"""

    json_dump = f"""with open(r'{save_filename}.json','w') as f:
    json.dump(meta_dict,f)
""" 

    final_val = """import mpld3
import json
"""
    final_val += new_val
    print(new_val)
    print(value)
    print(final_val)
    if(value is not new_val):
        print("test")
        if(globals()["visualizationMode"] == False):
            print("\n\nfailed")
            with open("output.txt", "w") as op:
                op.write("Text Mode enabled. Please enable visualization mode")
            return "failed"
        globals()['count'] += 1
        final_val += metadata
        final_val += json_dump
        final_val += "\nplt.close()"
        
        with open('code/temp.py', "w") as f:
            f.write(final_val)

        with open('code/temp.py', "r") as f:
            try:
                exec(f.read())
                return save_filename[7:]
            except Exception as e:
                print(e)
                return ""
    else:
        if(globals()["visualizationMode"]):
            with open("output.txt", "w") as op:
                op.write("Visualization Mode enabled. Please change to text mode")    
                return "failed";
        with open('code/temp.py', "w") as f:
            f.write(value)
        with open("output.txt", "w") as op:
            subprocess.call(["python", "code/temp.py"], stdout=op);
            print("done")
        return "text"
    # -- Python Code generated -- #

    # We can just write it into a file and use subprocess()/exec()

    


def queryResponseGenerator(query : str, filename : str) -> str:
    csv = pd.read_csv(filename)
    example = csv.head(2)
    output = chain.run(query = query, filename = filename, csv_example=example)
    return parse_and_run_output(output)
    

class QueryProcessor(Resource):
    def post(self):
        data = request.get_json()['input'] 
        name = queryResponseGenerator(data, 'Housing.csv')
        new_dict = {}
        data += "\n(visualization enabled)" if globals()["visualizationMode"] else "\n(text)"
        new_dict['user'] = data
        if name == "text":
            temp = PromptTemplate.from_template(summarize_prompt)
            with open("output.txt", "r") as op:
                det = op.read()
                print(det)
                new_dict['bot'] = chain.llm.predict(temp.format(information=det))
            globals()['chat_messages'].append(new_dict)
            return jsonify({'answer': "Text output"})
        elif name == "":
            new_dict['bot'] = "Server Error : Please try again later"
            globals()['chat_messages'].append(new_dict)
            return jsonify({'answer' : "500: Server Error"})
        elif name == "failed":
            with open("output.txt", "r") as op:
                new_dict['bot'] = op.read() 
                globals()['chat_messages'].append(new_dict)
                return jsonify({'answer' : "Wrong mode"})

        else:
            new_url_for = url_for('static', filename=f"{name}.html")
            with open(f"static/{name}.json","r") as f:
                data = f.read()
                data_json = json.loads(data)
                new_dict['bot'] = f"<iframe src='{new_url_for}' id='img-message' style='border:none;height:{data_json['height']};width:{data_json['width']};'></iframe>"
            #new_dict['bot'] = f"<img src='{new_url_for}' id='img-message'>"
           
            globals()['chat_messages'].append(new_dict)
            return jsonify({'answer' : name})

api.add_resource(QueryProcessor, '/chat')

@app.route('/')
def redirectPage():
    return redirect('/text')

@app.route('/text', methods=['POST','GET'])
def textPage():
    globals()["visualizationMode"] = False
    return render_template('index.html', messages=globals()["chat_messages"], mode = visualizationMode)

@app.route('/visualization', methods=['POST','GET'])
def visualizePage():
    globals()["visualizationMode"] = True
    return render_template('index.html', messages=globals()["chat_messages"], mode = visualizationMode)


if __name__ == '__main__':
    app.run(debug=False)


# class Reader:
#     csv_file : str
#     df : pd.DataFrame
#     def __init__(self, csv):
#         self.csv_file = csv
#     def read(self,csv : str = "") -> pd.DataFrame:
#         if csv != "":
#             self.csv_file = csv
#         self.df = pd.read_csv(self.csv_file)            
#         return self.df
 