import re
import json
import requests
import yaml, os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# This file contains methods that the LLM model will use when invoking the Agent tools


class AgentUtils:
    def __init__(self, llm, client):
        self.llm = llm
        self.client = client

    def extraction(self, input):
        """
         Use Langchain chain to let LLM parse the scientific conference information from the input text.

         Args:
             input (str): The extracted object, in this project usually the reference part a the paper

         Returns:
             str: The extraction results include the original citations, possible conference titles and short name.
        """

        template = os.path.join(os.path.dirname(__file__), 'templates.yaml').replace("\\", "/")

        with open(template, 'r', encoding='utf-8') as file:
            templates = yaml.safe_load(file)
            fact_extraction_template = templates['fact_extraction_template']

        fact_extraction_prompt = PromptTemplate(
            input_variables=["text_input"],
            template= fact_extraction_template
        )
        fact_extraction_chain = LLMChain(llm=self.llm, prompt=fact_extraction_prompt)
        facts = fact_extraction_chain.run(input)
        return facts


    def qid_query(self, qid):
        """
         Query the detailed Wikidata metadata of the conference according to the extracted QID.

         Args:
             qid (str): Extracted QID

         Returns:
             str: Corresponding metadata in Wikidata, including title, short name, start/end date, location and conference website.
        """

        query = """
        SELECT ?QID ?conferenceLabel ?startDate ?endDate ?locationLabel ?officialWebsite
        WHERE {
          VALUES ?conference {wd:""" + qid + """}
          OPTIONAL { ?conference wdt:P580 ?startDate. }
          OPTIONAL { ?conference wdt:P582 ?endDate. }
          OPTIONAL { ?conference wdt:P276 ?location. }
          OPTIONAL { ?conference wdt:P856 ?officialWebsite. }
          BIND(REPLACE(STR(?conference), ".*/(Q[0-9]+)$", "$1") AS ?QID)

          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        """

        url = "https://query.wikidata.org/sparql"
        params = {
            "query": query,
            "format": "json"
        }
        response = requests.get(url, params=params)
        data = response.json()
        results = data.get("results", {}).get("bindings", [])

        conference_metadata = []
        for result in results:
            qid = result["QID"]["value"]
            title = result.get("conferenceLabel", {}).get("value", "N/A")
            startDate = re.findall(r'\d{4}-\d{2}-\d{2}', result.get("startDate", {}).get("value", ""))[0]
            endDate = re.findall(r'\d{4}-\d{2}-\d{2}', result.get("endDate", {}).get("value", ""))[0]
            location = result.get("locationLabel", {}).get("value", "N/A")
            officialWebsite = result.get("officialWebsite", {}).get("value", "N/A")

            conference = {
                "Qid": qid,
                "Title": title,
                "StartDate": startDate,
                "EndDate": endDate,
                "Location": location,
                "OfficialWebsite": officialWebsite
            }
            conference_metadata.append(conference)
        return conference_metadata


    def weaviate_query_input(self, title):
        """
         Use Langchain chain to let LLM query whether the input conference is stored in the Weaviate VS, and if so, obtain detailed information based on QID.

         Args:
             title (str): Conference title for query in Weaviate VS.

         Returns:
             str: The query result (whether there is a matching conference in the database), and if so, the detailed metadata in Wikidata.
        """

        response = (
            self.client.query
            .get("Conference", ["qid", "title", 'shortName'])
            .with_limit(5)
            .with_near_text({
                "concepts": title
            })
            .do()
        )
        result = json.dumps(response)
        merged = "extrationTitle:" + title + '/queryResult:' + result

        template = os.path.join(os.path.dirname(__file__), 'templates.yaml').replace("\\", "/")
        with open(template, 'r', encoding='utf-8') as file:
            templates = yaml.safe_load(file)

        weaviate_query_input_template = templates['weaviate_query_input_template_new']

        query_Input_prompt = PromptTemplate(
            input_variables=["text_input"],
            template=weaviate_query_input_template
        )

        weaviateQuery_chain = LLMChain(llm=self.llm, prompt=query_Input_prompt)

        analysis_result = weaviateQuery_chain.run(merged)

        qidQuery_result_str = "[]"
        qid_match = re.search(r'Q[0-9]+', analysis_result)
        qid = qid_match.group() if qid_match else None

        if qid:
            qidQuery_result = self.qid_query(qid)  # Call qid_query method with the extracted QID
            qidQuery_result_str = json.dumps(qidQuery_result)

        return analysis_result + "Detailed metadata " + qidQuery_result_str
