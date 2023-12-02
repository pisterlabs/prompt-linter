import json
import argparse
import datetime

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from SPARQLWrapper import SPARQLWrapper

import utilities.sparql_functions as sparql_f
import utilities.eval_metrics as eval_metrics
import utilities.llm_tasks_prompts as llm_tasks
import utilities.entity_link as el
import utilities.data_io as dio
import utilities.nlp_tasks as nlp_tasks
import utilities.emb_tasks as emb_tasks
from utilities.timeout import time_limit, TimeoutException

# Create a list of QA pairs
qa_pairs = dict()

start_time = datetime.datetime.now()

qa_pairs['start_time'] = str(start_time)

sparql_wd = SPARQLWrapper("https://query.wikidata.org/sparql")
# sparql_dbp = SPARQLWrapper("http://dbpedia.org/sparql")

# Add arguments to the command line
parser = argparse.ArgumentParser(description='Run the combined LLM-KG on a dataset.')
parser.add_argument('-d', '--dataset', help='The dataset to run the LLM-KG on.', required=True)
args = parser.parse_args()

dataset_path = ""
json_output_path = ""

if args.dataset == "geography":
    dataset_path = "../output/qa_sets_basic_llm_geography1695750702.298964.json"
    # json_output_path = f"../output/qa_sets_llm_kg_geography_wd_{start_time.timestamp()}.json"
    json_output_path = f"../output/qa_sets_geography_evaluation.json"
elif args.dataset == "government_and_politics":
    dataset_path = "../output/qa_sets_basic_llm_government_and_politics1695751141.861084.json"
    # json_output_path = f"../output/qa_sets_llm_kg_government_and_politics_wd_{start_time.timestamp()}.json"
    json_output_path = f"../output/qa_sets_government_and_politics_evaluation.json"
elif args.dataset == "miscellaneous":
    dataset_path = "../output/qa_sets_basic_llm_miscellaneous_1695753134.334187.json"
    # json_output_path = f"../output/qa_sets_llm_kg_miscellaneous_wd_{start_time.timestamp()}.json"
    json_output_path = f"../output/qa_sets_miscellaneous_evaluation.json"
else:
    print("Invalid dataset.")
    exit()

# Load the data
# data = dio.read_data(dataset_path)
data = json.load(open(dataset_path))

# Create the language model
llm = OpenAI(temperature=0)

num_correct = 0

# Generate a response for each question
for i in range(len(data) - 3):
    item = data[str(i)]
    question = item['question']
    response = item['initial_response']

    print("Q:", question)
    print("A:", response, "\n")

    qa_pairs[i] = dict()
    qa_pairs[i]["question"] = question
    qa_pairs[i]["choices"] = item['choices']
    qa_pairs[i]["initial_response"] = response.strip()

    new_response = ""
    try:
        with time_limit(180):
            # Use the LLM to extract entities from the question
            question_entities = llm_tasks.extract_entities(question)
            print("Q entities:", question_entities)

            # Use the LLM to extract entities from the response
            response_entities = llm_tasks.extract_entities(response.strip())
            print("A entities:", response_entities)

            # Combine the entities from the question and response
            entity_names = list(dict.fromkeys(question_entities + response_entities))
            print("All entities:", entity_names)

            qa_pairs[i]["question_entity_names"] = question_entities
            qa_pairs[i]["response_entity_names"] = response_entities

            # Feed question, LLM response, and entities into LLM and extract knowledge graph facts from the response
            triples_names = llm_tasks.extract_kg_facts_given_entities(f"{question} {response.strip()}", entity_names)
            triples_names = nlp_tasks.remove_stopwords_from_triples(triples_names)
            print("Triples:", triples_names)
            print()

            qa_pairs[i]["extracted_triples"] = triples_names

            name_uri_map = dict()
            uri_name_map = dict()
            uri_label_desc_map = dict()

            extr_triples_uris = []

            # Convert the triples to a list of URIs
            for triple in triples_names:
                context_str = f"{triple[0]} {triple[1]} {triple[2]}"

                uris = []  # Used to construct the triple as a tuple of URIs

                # Get the URI for each component of the triple
                for j, component in enumerate(triple):
                    # Use REST API to retrieve the URI of the entity/property
                    if j == 1:
                        if (component, 'rel') not in name_uri_map:
                            uri, desc, label = el.fetch_uri_wikidata(component, context_str, ent_type='property')
                            name_uri_map[(component, 'rel')] = uri
                            uri_name_map[uri] = component
                            uri_label_desc_map[uri] = (label, desc)
                        else:
                            uri = name_uri_map[(component, 'rel')]
                    else:
                        if (component, 'ent') not in name_uri_map:
                            uri, desc, label = el.fetch_uri_wikidata(component, context_str)
                            name_uri_map[(component, 'ent')] = uri
                            uri_name_map[uri] = component
                            uri_label_desc_map[uri] = (label, desc)
                        else:
                            uri = name_uri_map[(component, 'ent')]
                    uris.append(uri)
                print()

                extr_triples_uris.append(tuple(uris))

            # Print triples as tuple of URIs
            print("Extracted triples URIs:", extr_triples_uris, '\n')
            qa_pairs[i]["extracted_triples_uris"] = extr_triples_uris

            # Get the URIs for the entities in the question
            question_ent_uris = []
            for ent_name in question_entities:
                if (ent_name, 'ent') not in name_uri_map:
                    ent_uri, ent_desc, ent_label = el.fetch_uri_wikidata(ent_name, ent_name)
                    name_uri_map[(ent_name, 'ent')] = ent_uri
                    uri_name_map[ent_uri] = ent_name
                    uri_label_desc_map[ent_uri] = (ent_label, ent_desc)
                question_ent_uris.append(name_uri_map[(ent_name, 'ent')])

            print("Question entities URIs:", question_ent_uris)
            print()

            qa_pairs[i]["uri_name_map"] = uri_name_map
            qa_pairs[i]["uri_label_desc_map"] = uri_label_desc_map

            truth_score = 0
            true_facts_uris = []
            true_facts_names = []
            true_entities_uris = set()

            # For each triple, perform a SPARQL query to verify the truthfulness
            for s, p, o in extr_triples_uris:
                print("Triple:", s, p, o)

                # Extend the description of the subject, predicate and object
                s_label, s_desc = uri_label_desc_map[s]
                p_label, p_desc = uri_label_desc_map[p]
                if p_desc is None:
                    p_ext_desc = p_label
                else:
                    p_ext_desc = p_label + " is " + p_desc
                o_label, o_desc = uri_label_desc_map[o]
                if o_desc is None:
                    o_ext_desc = o_label
                else:
                    o_ext_desc = o_label + " is " + o_desc

                p_uri_label_trips_forw, p_uri_label_trips_back, o_uri_label_trips = \
                    sparql_f.get_sparql_results_wikidata_described(s, p, o)

                print("Predicate URI label desc triplets:", p_uri_label_trips_forw)
                print("Object URI label desc triplets:", o_uri_label_trips)

                best_sim_score = 0
                best_fact_with_names = None
                best_fact_with_uris = None

                for other_p_uri, other_p_label, other_p_desc in p_uri_label_trips_forw:
                    print("Predicate URI-label pair:", other_p_uri, other_p_label)
                    if isinstance(other_p_desc, str):
                        sim_score = emb_tasks.calculate_squared_cos_sim(p_ext_desc, other_p_label + " is " + other_p_desc)
                    else:
                        sim_score = emb_tasks.calculate_squared_cos_sim(p_ext_desc, other_p_label)
                    print("Predicate similarity score:", sim_score)
                    if sim_score > best_sim_score:
                        best_sim_score = sim_score
                        best_fact_with_names = (s_label, other_p_label, o_label)
                        best_fact_with_uris = (s, other_p_uri, o)

                for other_p_uri, other_p_label, other_p_desc in p_uri_label_trips_back:
                    print("Predicate URI-label pair:", other_p_uri, other_p_label)
                    if isinstance(other_p_desc, str):
                        sim_score = emb_tasks.calculate_squared_cos_sim(p_ext_desc, other_p_label + " is " + other_p_desc)
                    else:
                        sim_score = emb_tasks.calculate_squared_cos_sim(p_ext_desc, other_p_label)
                    print("Predicate similarity score:", sim_score)
                    if sim_score > best_sim_score:
                        best_sim_score = sim_score
                        best_fact_with_names = (o_label, other_p_label, s_label)
                        best_fact_with_uris = (o, other_p_uri, s)

                for other_o_uri, other_o_label, other_o_desc in o_uri_label_trips:
                    print("Object URI-label pair:", other_o_uri, other_o_label)
                    if isinstance(other_o_desc, str):
                        sim_score = emb_tasks.calculate_squared_cos_sim(o_ext_desc, other_o_label + " is " + other_o_desc)
                    else:
                        sim_score = emb_tasks.calculate_squared_cos_sim(o_ext_desc, other_o_label)
                    print("Object similarity score:", sim_score)
                    if sim_score > best_sim_score:
                        best_sim_score = sim_score
                        best_fact_with_names = (s_label, p_label, other_o_label)
                        best_fact_with_uris = (s, p, other_o_uri)

                print("Best similarity score:", best_sim_score)
                print("Best fact with names:", best_fact_with_names)
                print("Best fact with URIs:", best_fact_with_uris)

                if best_sim_score > 0:
                    truth_score += best_sim_score
                    true_facts_uris.append(best_fact_with_uris)
                    true_facts_names.append(best_fact_with_names)
                    true_entities_uris.add(best_fact_with_uris[0])
                    true_entities_uris.add(best_fact_with_uris[2])

                    uri_name_map[best_fact_with_uris[0]] = best_fact_with_names[0]
                    uri_name_map[best_fact_with_uris[1]] = best_fact_with_names[1]
                    uri_name_map[best_fact_with_uris[2]] = best_fact_with_names[2]

            print("Truth Score:", truth_score)

            if len(extr_triples_uris) > 0:
                frac_true = truth_score / len(extr_triples_uris)
            else:
                frac_true = 0
            print("Simple measure of truthfulness:", frac_true)

            print("True facts names:", true_facts_names)

            qa_pairs[i]["truth_score"] = truth_score
            qa_pairs[i]["% true"] = frac_true

            qa_pairs[i]["true_facts_names"] = true_facts_names
            qa_pairs[i]["true_facts_uris"] = true_facts_uris

            # Calculate the number of linked entities
            linked_entities = set()
            for s, p, o in extr_triples_uris:
                if s.startswith("wd:"):
                    linked_entities.add(s)
                if o.startswith("wd:"):
                    linked_entities.add(o)
            print("Linked Entities:", linked_entities)
            print("Number of Linked Entities:", len(linked_entities))

            qa_pairs[i]["linked_entities"] = list(linked_entities)

            # Evaluate the truthfulness of the response
            eval_score = eval_metrics.simple_evaluation_using_similarity(entity_names, linked_entities,
                                                                         triples_names, truth_score)
            print("Evaluation Score:", eval_score)
            print()

            qa_pairs[i]["evaluation_score"] = eval_score

    except TimeoutException:
        print("Timeout Exception")
        new_response = response.strip()
        qa_pairs[i]["timeout"] = True
    except Exception as exc:
        print("Error:", exc)
        new_response = response.strip()
        qa_pairs[i]["error"] = str(exc)

    with open(json_output_path, "w") as f:
        json.dump(qa_pairs, f, indent=4)
