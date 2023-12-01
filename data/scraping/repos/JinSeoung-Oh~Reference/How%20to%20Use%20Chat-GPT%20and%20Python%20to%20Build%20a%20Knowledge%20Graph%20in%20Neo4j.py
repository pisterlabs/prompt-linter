# from https://towardsdatascience.com/how-to-use-chat-gpt-and-python-to-build-a-knowledge-graph-in-neo4j-based-on-your-own-articles-c622bc4e2eaa

## After install openai and Neo4j

## write prompt_input.py
# Ex)
entities = ["Mathematical entity", "Person", "Location", "Animal", "Activity", "Programming language", "Equation", "Date", "Shape", "Property", "Mathematical expression", "Profession", "Time period", "Mathematical subject", "Mathematical concept", "Discipline", "Mathematical theorem", "Physical entity", "Physics subject", "Physics"]
relationships = ["IS", "ARE", "WAS", "EQUIVALENT_TO", "CONTAINS", "PROPOSED", "PARTICIPATED_IN", "SOLVED", "RELATED_TO", "CORRESPONDS_TO", "HAS_PROPERTY", "REPRESENTS", "IS_USED_IN", "DISCOVERED", "FOUND", "IS_SOLUTION_TO", "PROVED", "LIVED_IN", "LIKED", "BORN_IN", "CONTRIBUTED_TO", "IMPLIES", "DESCRIBES", "DEVELOPED", "HAS_PROPERTY", "USED_FOR"]

prompt = f"""You are a mathematician and a scientist helping us extract relevant information from articles about mathematics. 
The task is to extract as many relevant relationships between entities to mathematics, physics, or history and science in general as possible.
The entities should include all persons, mathematical entities, locations etc. 
Specifically, the only entity tags you may use are:
{', '.join(entities)}.
The only relationships you may use are:
{', '.join(relationships)}
As an example, if the text is "Euler was located in Sankt Petersburg in the 17 hundreds", the output should have the following format: Euler: Person, LIVED_IN, Skt. Petersburg: Location 
If we have "In 1859, Riemann proved Theorem A", then as an output you should return Riemann: Person, PROVED, Theorem A: Mathematical theorem
I am only interested in the relationships in the above format and you can only use what you find in the text provided. Also, you should not provide relationships already found and you should choose less than 100 relationships and the most important ones.
You should only take the most important relationships as the aim is to build a knowledge graph. Rather a few but contextual meaningful than many nonsensical. 
Moreover, you should only tag entities with one of the allowed tags if it truly fits that category and I am only interested in general entities such as "Shape HAS Area" rather than "Shape HAS Area 1".
The input text is the following:
"""

## connect.py
import os
import openai
from prompt_input import prompt

openai.api_key = "<Your API key goes here>"

def process_gpt4(text):
    """This function prompts the gpt-4 model and returns the output"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "user", "content": prompt + text},
        ],
    )

    result = response['choices'][0]['message']['content']

    return result

## extract_text_from_html.py 
from bs4 import BeautifulSoup


def extract_text_from_html(html_content):
    """This function extracts the text from the articles"""
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.extract()

    article_tag = soup.find('article')
    if article_tag:
        return " ".join(article_tag.stripped_strings)

## preprocess.py
def text_to_batches(s, batch_size=2000):
    words = s.split()
    batches = []
    
    for i in range(0, len(words), batch_size):
        batch = ' '.join(words[i:i+batch_size])
        batches.append(batch)
        
    return batches

## process_articels.py
import os
from tqdm import tqdm
from connect import process_gpt4
from extract_text import extract_text_from_html
from preprocess import text_to_batches

base_path = 'raw'
processed_articles = os.listdir('data')

for file_name in tqdm(os.listdir(base_path)):

    title = ' '.join(file_name.split('_')[-1].split('-')[:-1])
    if f'results_{title}.txt' in processed_articles:
        continue

    results = ''
    with open(os.path.join(base_path, file_name), 'r', encoding='utf-8') as f:
        content = f.read()
        extraction = extract_text_from_html(content)
        batches = text_to_batches(extraction)
        for batch in batches:
            gpt_results = process_gpt4(batch)
            results += gpt_results

    with open(f'data/results_{title}.txt', 'w', encoding='utf-8') as results_file:
         results_file.write(results)
        
    with open(f'cleaned/cleaned_{title}.txt', 'w', encoding='utf-8') as cleaned_file:
        cleaned_file.write(extraction)

## Building the knowledge graph
from neo4j import GraphDatabase


class LoadGraphData:
    def __init__(self, username, password, uri):
        self.username = username
        self.password = password
        self.uri = uri
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def create(self, query):
        with self.driver.session() as graphDB_Session:
            return graphDB_Session.run(query)

    def set_max_nodes(self, number):
        query = f":config initialNodeDisplay: {number}"
        with self.driver.session() as graphDB_Session:
            return graphDB_Session.run(query)

    def delete_graph(self):
        delete = "MATCH (n) DETACH DELETE n"
        with self.driver.session() as graphDB_Session:
            graphDB_Session.run(delete)

    @staticmethod
    def do_cypher_tx(tx, cypher):
        result = tx.run(cypher)
        values = []
        for record in result:
            values.append(record.values())
        return values

    def work_with_data(self, query):
        with self.driver.session() as session:
            values = session.read_transaction(self.do_cypher_tx, query)
        return values 

## make graph
import os
import re
from prompt_input import entities, relationships
from loader import LoadGraphData
from tqdm import tqdm


def create_relationships(loader, title, e1, l1, e2, l2, R):
    query = f'MERGE (:Article {{name: "{title}"}})\
            MERGE (:{l1} {{name: "{e1}"}})\
            MERGE (:{l2} {{name: "{e2}"}})'
    loader.create(query)

    query = f'MATCH (t:Article {{name: "{title}"}})\
            MATCH (a:{l1} {{name: "{e1}"}})\
            MATCH (b:{l2} {{name: "{e2}"}})\
            MERGE (a)-[:{R}]->(b)\
            MERGE (a)-[:IN_ARTICLE]->(t)\
            MERGE (b)-[:IN_ARTICLE]->(t)'
    loader.create(query)



def make_graph(source, cleaned):
    loader = LoadGraphData("neo4j", "<password>", "bolt://localhost:7687")
    loader.delete_graph()

    history = []
    for results in tqdm(os.listdir(source)):
        with open(os.path.join(source, results)) as r:
            content = r.read()
            lines = content.split('\n')
        if len(lines) < 10:
            continue

        with open(os.path.join(cleaned, 'cleaned_' + '_'.join(results.split('_')[1:]))) as c:
            cleaned_content = c.read()

        for line in lines:
            line = re.sub('^\d+\.', '', line).strip()
            splitted = line.split(',')
            if len(splitted) == 3:
                A = splitted[0]
                R = splitted[1].strip()
                B = splitted[2]

                if not ':' in A or not ':' in B:
                    continue

                e1, l1 = A.split(':')[0], A.split(':')[1]
                e2, l2 = B.split(':')[0], B.split(':')[1]
                e1, e2, l1, l2 = e1.strip(), e2.strip(), l1.strip(), l2.strip()

                if e1.lower() not in cleaned_content.lower() or e2.lower() not in cleaned_content.lower():
                    continue

                if l1 == 'Person':
                    for subname in  e1.split()[::-1]:
                        if subname[0].upper() == subname[0]:
                            e1 = subname
                            break
                
                if l2 == 'Person':
                    for subname in  e2.split()[::-1]:
                        if subname[0].upper() == subname[0]:
                            e2 = subname
                            break
                
                if R == R.upper() and R in relationships and l1 in entities and l2 in entities and len(e1.split()) < 5 and len(e1) > 1 and len(e2.split()) < 5 and len(e2) > 1 and e1 != e2:
                    if line not in history:
                        history.append(line)

                        l1 = l1.replace(" ", "_")
                        l2 = l2.replace(" ", "_")
                        e1 = e1.replace('"', '')
                        e2 = e2.replace('"', '')
                        title = results.split('.')[0].replace(' ', '_')
                        title = '_'.join(title.split('_')[1:])

                        create_relationships(loader=loader, title=title, e1=e1, l1=l1, e2=e2, l2=l2, R=R)

## main
from make_graph import make_graph

make_graph(source='data', cleaned='cleaned')
