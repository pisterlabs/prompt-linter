'''
This module contains the routes for TL;DR.
'''
import openai
import os
from flask import Blueprint, request, render_template, jsonify
import pickle
import sys

# -- Setting up the utils path module --
sys.path.append('utils')

tldr = Blueprint('tldr', __name__)
openai.api_key = os.environ.get("OPENAI_API_KEY")
print(os.environ.get("OPENAI_API_KEY"))


@tldr.route('/tldr')
def tldr_index():
    return render_template("/tldr.html")


@tldr.route('/tldr/search', methods=['GET'])
def tldr_search():
    user_input = request.args.get(
        'user_input') if request.method == 'GET' else request.form['user_input']
    # messages = [{"role": "user", "content": user_input}]
    if user_input == "":
        gene = "CESA1"
    else:
        gene = user_input.upper()
    query = '''
        You are a plant molecular biology expert and systems biologist. Consider the following data about the %s. Organize these facts into a comprehensive summary structured into the following categories:

        1) Enumerate all the functions that the %s performs based on the data.
        2) Identify all sub-cellular compartments (e.g, nucleus, cytosol, plasma membrane) where %s is present.
        3) Specify all the sites (cells, tissues, organs) where the %s is active.
        4) List all the Gene Ontology (GO) terms stated for %s. If none are stated, propose possible ones based on the data.
        5) Based on the data, list all the genes that %s interacts (e.g., interacts, binds, is in complex with) with.
        6) Mention all the genes that %s regulates, maintains or affectes.
        7) Detail all the genes that regulates, maintains or affectes %s according to the data.
        An entity refers to either a gene, molecule, compartment, stress, cell type, organ, or other related terms
        For each point, if a entity is said to "interact with", "maintain", "enhance", "repress", "regulate" or "activate" another entity, consider this as an interaction. When a entity "produces" or "encodes for" a substance, consider this as a function of the entity. If an entity is "found in" a particular site, consider this as a place where the entity is expressed(gene) or located(others).
        IMPORTANT Tag the CORRECT source behind each statement in parenthesis (e.g., (10024464)).
        Given your prior knowledge on %s in plant biology context, Please phrase it in a coherent and plesant fashion.
        Please start the reply with "Comprehensive Summary of %s"
        Please start with the categorical question; followed by the answer
        Please add this statement at the end "Please note that the sources are provided after each statement."
        Please make sure all the information give has been covered
        ''' % (gene, gene, gene, gene, gene, gene, gene, gene, gene, gene)

    save = ['source\tassociation type\ttarget\tsource\n']
    limit = 500
    warning = ""
    with open("allDic3", "rb") as file:
        allDic3 = pickle.load(file)
    for k, v in allDic3[gene[0]][gene[1]].items():
        if gene == k:
            save[0] += v
            if save[0].count("\n") > limit:
                warning = '''Network of %s is extensive, only the first %s entities that interact with %s will be displayed''' % (
                    gene, limit, gene)
                parts = save[0].split("\n", limit)
                shorten = "\n".join(parts[:limit])
                save[0] = shorten
            break
    print("query")
    messages = [{"role": "user", "content": save[0]+query}]
    print("messages", messages)
    output = openai.ChatCompletion.create(model="gpt-4", messages=messages)
    # v = open('search.txt', 'w')
    # v.writelines(save)
    # v.close()
    # loader = UnstructuredTSVLoader(
    #     file_path="search.txt", mode="elements"
    # )
    # loader.load()
    # index = VectorstoreIndexCreator().from_loaders([loader])
    # output = index.query(query, llm=ChatOpenAI(
    #     temperature=0.05, model="gpt-4"))
    print(output)
    return jsonify(content=output.choices[0].message.content, warning=warning)
