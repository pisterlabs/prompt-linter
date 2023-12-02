import os
import json
import csv
import re

import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from SPARQLWrapper import SPARQLWrapper, JSON
import spacy

# OpenAI API key
openai_api_key_file = open("../openai_api_key.txt", "r")
os.environ["OPENAI_API_KEY"] = openai_api_key_file.read().strip()
openai_api_key_file.close()

sparql_db = SPARQLWrapper("http://dbpedia.org/sparql")


def perform_sparql_query(query):
    """
    Perform a SPARQL query
    :param query: The SPARQL query
    :return: Output of the query
    """
    sparql_db.setQuery(query)
    sparql_db.setReturnFormat(JSON)
    results = sparql_db.query().convert()
    return results


# Load the data
file = open("../data/mmlu_test/high_school_geography_test.csv", "r")
csv_reader = csv.reader(file, delimiter=',')
data = []
for row in csv_reader:
    data.append({"question_text": row[0], "choices": row[1:-1], "correct_answer": row[-1]})

# Create the language model
llm = OpenAI(temperature=0)

# Create NLP model for extracting entities
nlp = spacy.blank("en")
nlp.add_pipe("dbpedia_spotlight")

# Create a list of QA pairs
# qa_pairs = dict()

num_correct = 0

# Generate a response for each question
for ident, item in enumerate(data[:5]):  # 66:68  # 85:87  # 189:190
    question = item['question_text']
    response = llm.predict(question)

    print("Q:", question)
    print("A:", response.strip(), "\n")

    # qa_pairs[ident] = dict()
    # qa_pairs[ident]["question"] = question
    # qa_pairs[ident]["response"] = response.strip()

    extraction_output = nlp(question + " " + response.strip())

    # print("Extraction Output:", extraction_output)
    print("Entities:", extraction_output.ents)
    # print([(ent.text, ent.kb_id_, ent._.dbpedia_raw_result['@similarityScore']) for ent in extraction_output.ents])
    print()

    sparql_results = []

    entities = list(set([ent.kb_id_ for ent in extraction_output.ents]))

    ents_aliases = dict()
    for ent in extraction_output.ents:
        if ent.kb_id_ not in ents_aliases:
            ents_aliases[ent.kb_id_] = []
        ents_aliases[ent.kb_id_].append(ent.text)

    for i, ent1 in enumerate(entities):
        '''
        subj_results = perform_sparql_query(f"""
               SELECT ?pred ?obj
                WHERE {{
                    <{ent1}> ?pred ?obj.
                    FILTER(!isLiteral(?obj) || lang(?obj) = "" || langMatches(lang(?obj), "EN"))
                }}
            """)

        preds_objs = [(result['pred']['value'], result['obj']['value']) for result in subj_results['results']['bindings']]

        for pred, obj in preds_objs:
            sparql_results.append({"subject": ent1, "predicate": pred, "object": obj,
                                   "subject_aliases": ents_aliases[ent1]})

        obj_results = perform_sparql_query(f"""
               SELECT ?subj ?pred
                WHERE {{
                    ?subj ?pred <{ent1}>.
                }}
            """)

        subjs_preds = [(result['subj']['value'], result['pred']['value']) for result in obj_results['results']['bindings']]

        for subj, pred in subjs_preds:
            sparql_results.append({"subject": subj, "predicate": pred, "object": ent1,
                                   "object_aliases": ents_aliases[ent1]})
        '''

        for ent2 in entities[i + 1:]:
            if ent1 != ent2:
                results1 = perform_sparql_query(f"""
                    SELECT ?pred
                    WHERE {{
                        <{ent1}> ?pred <{ent2}>.
                    }}
                """)

                preds1 = [result['pred']['value'] for result in results1['results']['bindings']]

                if len(preds1) > 0:
                    # print(f"Relations between {ents_aliases[ent1][0]} and {ents_aliases[ent2][0]}:")
                    # print(preds1)
                    # print()

                    sparql_results.append({"subject": ent1, "object": ent2, "predicates": preds1,
                                           "subject_aliases": ents_aliases[ent1], "object_aliases": ents_aliases[ent2]})

                results2 = perform_sparql_query(f"""
                    SELECT ?pred
                    WHERE {{
                        <{ent2}> ?pred <{ent1}>.
                    }}
                """)

                preds2 = [result['pred']['value'] for result in results2['results']['bindings']]

                if len(preds2) > 0:
                    # print(f"Relations between {ents_aliases[ent2][0]} and {ents_aliases[ent1][0]}:")
                    # print(preds2)
                    # print()

                    sparql_results.append({"subject": ent2, "object": ent1, "predicates": preds2,
                                           "subject_aliases": ents_aliases[ent2], "object_aliases": ents_aliases[ent1]})

    '''
    context_string = ""

    for sparql_result in sparql_results:
        subject_alias = sparql_result.get("subject_aliases",
                                          [sparql_result["subject"].split("/")[-1].split("#")[-1]])[0]
        object_alias = sparql_result.get("object_aliases",
                                            [sparql_result["object"].split("/")[-1].split("#")[-1]])[0]
        predicate_alias = sparql_result["predicate"].split("/")[-1].split("#")[-1]
        context_string += f"{subject_alias} {predicate_alias} {object_alias} . "

    print("Context String:", context_string, "\n")
    '''

    context_string = ""

    for sparql_result in sparql_results:
        subject_alias = sparql_result["subject_aliases"][0]
        object_alias = sparql_result["object_aliases"][0]

        for predicate in sparql_result["predicates"]:
            predicate_alias = predicate.split("/")[-1].split("#")[-1].replace("_", " ")
            context_string += f"{subject_alias} {predicate_alias} {object_alias} . "

    print("Context String:", context_string, "\n")

    # Create a prompt template for the KG enriched LLM query
    # prompt = PromptTemplate(
    #     input_variables=["question", "context"],
    #     template="{question}\n Context: {context}"
    # )
    #
    # enriched_response = llm(prompt.format(question=question, context=context_string))

    # print(enriched_response)

    # Convert the list of choices to a string
    choices_text = "\n".join([str(i + 1) + ". " + choice for i, choice in enumerate(item["choices"])])
    print(f"Options:\n{choices_text}\n")

    # Create a prompt template for the KG enriched LLM query
    prompt = PromptTemplate(
        input_variables=["question", "choices", "context"],
        template="Output the numbered option for the following question: \
        {question}\nOptions:\n{choices}\nContext: {context}"
    )

    enriched_response = llm(prompt.format(question=question, choices=choices_text, context=context_string))
    print("Response:", enriched_response)

    # Convert the response to the numbered choice
    numbers = [int(num) for num in re.findall(r'\d+', response.strip().split(".")[0])]
    if len(numbers) == 0:
        numbered_output = 1
    else:
        numbered_output = numbers[-1]
    letter_output = chr(ord('A') + int(numbered_output) - 1)

    print("Generated answer:", letter_output)

    # Evaluate the response
    is_correct = letter_output == item['correct_answer']
    print("Correct answer:", item['correct_answer'])
    print("Correct:", is_correct, "\n")

    # Update the number of correct answers
    if letter_output == item['correct_answer']:
        num_correct += 1

    data[ident]["initial_response"] = response.strip()
    data[ident]["context"] = context_string
    data[ident]["llm_answer"] = letter_output
    data[ident]["llm_is_correct"] = is_correct

    # qa_pairs[ident]["llm_facts"] = llm_facts
    # qa_pairs[ident]["sparql_queries"] = sparql_queries

print("EM:", num_correct / len(data))

# Save the QA pairs in a JSON file
with open("../output/qa_sets_geography_kg.json", "w") as f:
    json.dump(data, f, indent=4)

