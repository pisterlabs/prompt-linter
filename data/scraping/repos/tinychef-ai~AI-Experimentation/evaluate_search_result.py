from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import json
import csv
import sys
import constants
import file_parser as fp
import unicodedata


llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301", temperature=0.0)

qr_prompt = PromptTemplate(
    input_variables=["query", "recipe_name"],
    template="""i want you to act as culinary expert and
    evaluate if recipe_name: {recipe_name} is a valid response for query: {query}.
    output the following fields in json response: 
    1. overall_match: "yes" if recipe_name is a valid response for above query,
        "no" otherwise
    2. overall_match_reason_code: reason behind overall_match response. select one of the following
        "cuisine" if cuisine match or mismatch is the main reason,
        "ingredient" if ingredient match or mismatch is the main reason,
        "meal_time" if meal time match or mismatch is the main reason,
        "meal_type" if meal type match or mismatch is the main reason,
        "diet" if diet match or mismatch is the main reason,
        "other" if match or mismatch is for any other reason
    3. overall_match_reason: reason behind overall_match response in 1 sentence.
    Do not output any additional explanation.
    """,
)

qr_chain = LLMChain(llm=llm, prompt=qr_prompt)


def evaluate_query_recipe(query, recipe):
    output = qr_chain.run({"query": query,
                           "recipe_name": recipe})
    try:
        json_obj = json.loads(output)
        json_obj["parse_success"] = True

        return json_obj
    except ValueError:
        return {"parse_success": False,
                "raw_string": output}


def evaluate_query_results(search_result_file, evaluation_result_file):
    with open(evaluation_result_file, "w") as output_file:
        line = 0
        with open(search_result_file) as input_file:
            reader = csv.reader(input_file)
            headers = next(reader, None)

            for row in reader:
                line = line + 1
                print("evaluating line " + str(line))
                query_eval_list = [row[0]]
                for i in range(1, constants.TOP_K + 1):
                    if i<len(row) and row[i] and len((row[i]).strip()) > 0:
                        query_eval = evaluate_query_recipe(row[0], row[i])
                        query_eval["recipe_name"] = row[i]
                        query_eval_list.append(query_eval)
                    else:
                        break

                output_file.write(json.dumps(query_eval_list))
                output_file.write("\n")


def increment_reason_code_count(reason_code_counts, reason_code):
    if reason_code not in reason_code_counts.keys():
        reason_code_counts[reason_code] = 0

    reason_code_counts[reason_code] = reason_code_counts[reason_code] + 1


def convert_query_results_to_csv_and_summarize(evaluation_result_file):
    response_count = [0] * constants.TOP_K
    parse_success_count = [0] * constants.TOP_K
    match_success_count = [0] * constants.TOP_K
    match_success_reason_count = []
    match_failure_reason_count = []
    atleast_one_match_success_count = 0

    for i in range(constants.TOP_K):
        match_success_reason_count.append({})
        match_failure_reason_count.append({})

    with open(evaluation_result_file) as input_file:
        writer = csv.writer(sys.stdout)
        writer.writerow(["query", "response_count", "atleast_one_match", "recipe_name_position",
                         "recipe_name", "overall_match", "overall_match_reason_code", "overall_match_reason"])

        for line in input_file:
            # line = line.strip()
            json_obj = json.loads(line)
            atleast_one_match = False

            for i in range(1, len(json_obj)):
                if json_obj[i]["parse_success"] and json_obj[i]["overall_match"] == "yes":
                    atleast_one_match = True

            if atleast_one_match:
                atleast_one_match_success_count = atleast_one_match_success_count + 1

            for i in range(1, len(json_obj)):
                response_count[i - 1] = response_count[i - 1] + 1
                if json_obj[i]["parse_success"]:
                    writer.writerow([json_obj[0], len(json_obj) - 1, atleast_one_match, i,
                                     json_obj[i]["recipe_name"],
                                     json_obj[i]["overall_match"],
                                     json_obj[i]["overall_match_reason_code"],
                                     json_obj[i]["overall_match_reason"]])
                    parse_success_count[i - 1] = parse_success_count[i - 1] + 1
                    match_reason_code = json_obj[i]["overall_match_reason_code"]
                    if json_obj[i]["overall_match"] == "yes":
                        match_success_count[i - 1] = match_success_count[i - 1] + 1
                        increment_reason_code_count(match_success_reason_count[i - 1], match_reason_code)
                    else:
                        increment_reason_code_count(match_failure_reason_count[i - 1], match_reason_code)

            if len(json_obj) == 1:
                writer.writerow([json_obj[0], 0, 0, None, None, None, None, None])

    print("atleast_one_match_success_count, " + str(atleast_one_match_success_count))
    print("response_count, " + json.dumps(response_count))
    print("parse_success_count, " + json.dumps(parse_success_count))
    print("match_success_count, " + json.dumps(match_success_count))
    for reason_code in match_success_reason_count[0].keys():
        print("match_success_count " + reason_code + ", ", end="")
        for i in range(constants.TOP_K):
            cnt = 0
            if reason_code in match_success_reason_count[i].keys():
                cnt = match_success_reason_count[i][reason_code]
            print(str(cnt) + ", ", end="")
        print("")

        print("match_failure_count " + reason_code + ", ", end="")
        for i in range(0, constants.TOP_K):
            cnt = 0
            if reason_code in match_failure_reason_count[i].keys():
                cnt = match_failure_reason_count[i][reason_code]
            print(str(cnt) + ", ", end="")
        print("")


def get_openai_valid_results(query_results,
                             openai_results):
    openai_valid_results = {}

    for query in query_results:
        valid_results = []
        for recipe_name in query_results[query]:
            if recipe_name in openai_results[query]:
                valid_results.append(recipe_name)
            else:
                break

            if len(valid_results) == constants.TOP_K:
                break

        openai_valid_results[query] = valid_results

    return openai_valid_results


def modify_evals_using_openai_results(openai_search_results_file,
                                      openai_search_result_evals_file,
                                      search_result_evals_file):
    query_results, openai_results = fp.parse_search_results(openai_search_results_file, True)
    openai_valid_results = get_openai_valid_results(query_results, openai_results)
    openai_search_result_evals = fp.parse_search_result_evals(openai_search_result_evals_file)

    with open(search_result_evals_file) as input_file:
        for line in input_file:
            json_obj = json.loads(line)
            query = json_obj[0].strip().lower()
            evals = [query]

            if query not in openai_search_result_evals.keys():
                continue

            for recipe_name in openai_valid_results[query]:
                evals.append(openai_search_result_evals[query][recipe_name])

            evals_dict = dict.fromkeys(openai_valid_results[query])
            for i in range(1, len(json_obj)):
                if len(evals) >= constants.TOP_K + 1:
                    break

                if json_obj[i]["recipe_name"].strip().lower() in evals_dict.keys():
                    continue

                evals.append(json_obj[i])

            print(json.dumps(evals))


def sample_search_results(search_result_file1, search_result_file2):
    search_results1, _ = fp.parse_search_results(search_result_file1)

    with open(search_result_file2) as input_file:
        reader = csv.reader(input_file)
        writer = csv.writer(sys.stdout)

        header = next(reader, None)
        writer.writerow(header)

        for row in reader:
            query = row[0].strip().lower()
            if query in search_results1 and (len(search_results1[query]) == 0 or
                                             "DEVANAGARI" not in unicodedata.name(search_results1[query][0][0])):
                writer.writerow(row)


def main():

    # evaluate_query_results("/Users/agaramit/Downloads/Search Results - Skill_Sheet1_sample.csv",
    #                        "/Users/agaramit/Downloads/Search Results - Skill_Sheet1_sample_eval.json")

    # evaluate_query_results("/Users/agaramit/Downloads/Search Results - Skill_app_searches_1_sample.csv",
    #                        "/Users/agaramit/Downloads/Search Results - Skill_app_searches_1_sample_eval.json")

    # evaluate_query_results("/Users/agaramit/Downloads/Search Results - Skill_app_searches_without_openai_sample.csv",
    #                        "/Users/agaramit/Downloads/Search Results - Skill_app_searches_without_openai_sample_eval.json")

    # evaluate_query_results("/Users/agaramit/Downloads/Search Results - openai_faiss_sample.csv",
    #                        "/Users/agaramit/Downloads/Search Results - openai_faiss_sample_eval.json")

    # evaluate_query_results("/Users/agaramit/Downloads/Search Results - openai_faiss4_sample.csv",
    #                        "/Users/agaramit/Downloads/Search Results - openai_faiss4_sample_eval.json")

    # evaluate_query_results("/Users/agaramit/Downloads/Search Results - openai_faiss5_sample.csv",
    #                        "/Users/agaramit/Downloads/Search Results - openai_faiss5_sample_eval.json")

    # evaluate_query_results("/Users/agaramit/Downloads/Search Results - openai_faiss5_1_sample.csv",
    #                        "/Users/agaramit/Downloads/Search Results - openai_faiss5_1_sample_eval.json")

    # evaluate_query_results("/Users/agaramit/Downloads/Search Results - solr_sample.csv",
    #                        "/Users/agaramit/Downloads/Search Results - solr_sample_eval.json")

    # evaluate_query_results("/Users/agaramit/Downloads/Search Results - solr_nomm_sample.csv",
    #                        "/Users/agaramit/Downloads/Search Results - solr_nomm_sample_eval2.json")

    evaluate_query_results("/Users/agaramit/Downloads/Search Results - openai_faiss5_expq1_sample.csv",
                           "/Users/agaramit/Downloads/Search Results - openai_faiss5_expq1_sample_eval.json")


    # convert_query_results_to_csv_and_summarize("/Users/agaramit/Downloads/Search Results - Skill_Sheet1_sample_eval.json")
    # convert_query_results_to_csv_and_summarize("/Users/agaramit/Downloads/Search Results - Skill_app_searches_1_sample_eval.json")


    # modify_evals_using_openai_results("/Users/agaramit/Downloads/Search Results - Skill_app_searches.csv",
    #                                   "/Users/agaramit/Downloads/Search Results - Skill_app_searches_1_sample_eval.json",
    #                                   "/Users/agaramit/Downloads/Search Results - openai_faiss5_expq1_sample_eval.json")

    # sample_search_results("/Users/agaramit/Downloads/Search Results - Skill_Sheet1_sample.csv",
    #                        "/Users/agaramit/Downloads/Search Results - Skill_app_searches_without_openai.csv")


if __name__ == "__main__":
    main()