from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import json
import csv

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301", temperature=0.0)

query_prompt = PromptTemplate(
    input_variables=["query"],
    template="""i want you to act as culinary expert and evaluate if 
    the query: {query} contains one or more valid search criteria to search for a recipe. 
    Valid search criteria list:
    'ingredient name, recipe name, cuisine name, meal type, diet type, cooking duration, calories, time of day, season, festival'.
    Ignore spelling errors while doing the above evaluation.
    
    output the following fields in json response: 
    1. query_valid: "yes" if query contains one or more valid search criteria to search for a recipe,
        "no" otherwise
    2. query_valid_reason: reason behind why the query is valid or not valid 
    
    Do not output any additional explanation.
    """,
)

query_chain = LLMChain(llm=llm, prompt=query_prompt)


def evaluate_query(query):
    output = query_chain.run(query)
    try:
        json_obj = json.loads(output)
        json_obj["parse_success"] = True

        return json_obj
    except ValueError:
        return {"parse_success": False,
                "raw_string": output}


def evaluate_queries(search_result_file, evaluation_result_file):
    with open(evaluation_result_file, "w") as output_file:
        line = 0
        with open(search_result_file) as input_file:
            reader = csv.reader(input_file)
            headers = next(reader, None)

            for row in reader:
                line = line + 1
                print("evaluating query " + str(line))
                query_eval = evaluate_query(row[0])
                query_eval["query"] = row[0]

                output_file.write(json.dumps(query_eval))
                output_file.write("\n")


evaluate_queries("/Users/agaramit/Downloads/Search Results - Skill_app_searches_1_sample.csv",
                    "/Users/agaramit/Downloads/query_eval.json")
