## EMBEDDINGS + ENTITY:

from memory import Memory
import spacy
from sklearn.neighbors import NearestNeighbors
import openai
from utils import load
import json
from rich import print
import guardrails as gd
from IPython import embed
spc = spacy.load("en_core_web_md")
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")


class EmbeddingEntityMemory(Memory):
    def __init__(self, source_text, k=3):
        super().__init__(source_text)
        self.lines = source_text.splitlines()
        # print("This was source text")
        # print(source_text)

        # EMBEDDINGS
        # Use spaCy to create Doc objects for all the lines in the file
        line_docs = [spc(line) for line in self.lines]

        # Create a matrix of document vectors
        X = [doc.vector for doc in line_docs]

        # Instantiate a NearestNeighbors object with k=3 (the number of neighbors to return)
        self.nn = NearestNeighbors(n_neighbors=k, algorithm='auto')

        # Fit the NearestNeighbors object to the document vectors
        self.nn.fit(X)

        # Initialize guardrails
        guard = gd.Guard.from_rail('entity_mapping.rail')

        # ENTITY
        entity_dict = {}
        curr_lines = ""
        i = 0

        process_count = 0
        # Fetch all relevant entities from every line
        # and add them to entity_dict
        line_chunk_size = 2
        for line in source_text.splitlines():
            if i < len(source_text) and i < line_chunk_size:
                curr_lines += line
                curr_lines += "\n"
                i += 1

            # Fetch all entities in curr_lines
            if i == line_chunk_size or i == len(source_text) - 1:
                # Wrap the OpenAI API call with the `guard` object
                toks = enc.encode(curr_lines)
                print(f"token count = {len(toks)}")
                raw_llm_output, validated_output = guard(
                    openai.Completion.create,
                    prompt_params={"curr_lines": curr_lines},
                    engine="text-davinci-003",
                    max_tokens=1000,
                    temperature=0.7,
                )
                with open(f"currlines_{process_count}.txt", "w") as f:
                    f.write(curr_lines)
                # print("This is it!!! raw llm output")
                # print(raw_llm_output)
                # validated_output = {}
                print("below is validated")
                print(validated_output)
                print(guard.state.most_recent_call.history[0].rich_group)
                # embed()


                # prompt = f"""
                #     Please provide a JSON with the format
                #     {{ "entity1": "sentences1", "entity2": "sentences2" }}
                #     of all relevant entities mentioned in the following conversation, along with all of their associated sentences, formatted as a proper JSON object. Make sure the last entry does NOT have a comma after it! Please include the speakers in the conversation as their own entities, and if they have personal information about themselves put all of those as the associated sentences. \n {curr_lines}
                # """
                # response = openai.Completion.create(
                #     engine="text-davinci-003",
                #     prompt=prompt,
                #     temperature=0,
                #     max_tokens=3000,
                #     stop=None,
                #     timeout=10
                # )
                # json_string = response.choices[0].text.replace('[', '').replace(']', '').strip()
                json_data = validated_output
                # print("Before")
                # print(json_string)
                # if json_string[-1] != '}':
                #     json_string += '}'
                # if json_string[0] != '{':
                #     json_string = '{' + json_string
                # print("This is the json string")
                # print(json_string)

                # # Find the last occurrence of the closing curly brace
                # last_brace_index = json_string.rfind("}")

                # Find the index of the first non-whitespace character before the closing curly brace
                # index = last_brace_index - 1
                # while index >= 0 and json_string[index].isspace():
                #     index -= 1
                # first_character_before_closing_brace = json_string[index]
                # if index >=0 and first_character_before_closing_brace == ",": # if the character before the closing brace is a comma
                #     # json_string = json_string[:index] + json_string[index+1:last_brace_index] + json_string[last_brace_index+1:]
                #     # new_str = str[:8] + str[9:]
                #     json_string = json_string[:index] + json_string[index+1:]

                # # if the charactr before the closing brace is not a comma and if is also not a " make sure you add the " before the closing brace
                # elif index >= 0 and first_character_before_closing_brace != "\"":
                #     json_string = json_string[:last_brace_index] + "\"" + json_string[last_brace_index:]

                # # Check if the character immediately before the closing curly brace is a comma
                # if index >= 0 and json_string[index] == "\"":
                #     pass
                # else:
                #     # Add the " character before the closing curly brace
                #     json_string = json_string[:last_brace_index] + "\"" + json_string[last_brace_index:]

                # # Find the last occurrence of the closing curly brace
                # last_brace_index = json_string.rfind("}")

                # # Find the index of the first non-whitespace character before the closing curly brace
                # index = last_brace_index - 1
                # while index >= 0 and json_string[index].isspace():
                #     index -= 1

                # if index >= 0 and json_string[index] == ",":
                #     print("We should remove the ,")
                #     json_string = json_string[:index] + json_string[index+1:last_brace_index] + json_string[last_brace_index+1:]
                #     print("Updated string")
                #     print(json_string)

                # Find the last occurrence of the closing curly brace
                # last_quote_index = json_string.rfind("\"")

                # Find the index of the first non-whitespace quote character before the closing curly brace
                # index = last_quote_index - 1
                # while index >= 0 and json_string[index].isspace():
                    # index -= 1

                # Check if the character immediately before the closing curly brace is a "
                # if index >= 0 afvgb_string)
                # if json_string[-1] != '}':
                #     json_string += '}'

                # json_data = json.loads(json_string)

                # Transform into intended format
                temp_entity_dict = {item["entity_name"]: item["sentences"] for item in list(validated_output.values())[0]}
                # temp_entity_dict = {}

                for key, value in temp_entity_dict.items():
                    uppercase_key = key.upper()
                    combined_value = ''.join(value)
                    if uppercase_key in entity_dict:
                        entity_dict[uppercase_key] += " " + combined_value
                    else:
                        entity_dict[uppercase_key] = combined_value
                curr_lines = ""
                i = 0

        self.entity_dict = entity_dict

    def query(self, query):

        # EMBEDDINGS
        # Use spaCy to create a Doc object for the new string
        new_doc = spc(query)

        # Find the indices of the k nearest neighbors to the new_doc
        distances, indices = self.nn.kneighbors([new_doc.vector])

        # Construct the closest embeddings
        closest_embeddings = ""
        for i in range(len(indices[0])):
            closest_embeddings += f"{self.lines[indices[0][i]]}\n"

        # ENTITY
        # Still use spacy in querying extract entities
        nlp = spacy.load("en_core_web_sm")
        context = ""

        # Find all the relevant entities in the query
        doc = nlp(query)
        for ent in doc.ents:
            print("This is an ent")
            print(ent.text)
            # Put existing entities in context
            if ent.text.upper() in self.entity_dict:
                context += self.entity_dict[ent.text.upper()] + "."

        # Use the closest embeddings to ask GPT to answer the question
        prompt = f"""You are a smart, knowledgeable, accurate AI with the following information:
            {closest_embeddings} {context}\n
            Please answer the following question: {query}
            """
        print("calling gpt3 in embeddings_topk.")
        print(prompt)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=250,
        )
        return response.choices[0].text.strip()

if __name__ == "__main__":
    source_text = load("test.txt")
    memory_test = EmbeddingEntityMemory(source_text, 3)

    query ="What is Mimi's favorite physical activity?"
    answer = memory_test.query(query)
    print(query)
    print(answer)