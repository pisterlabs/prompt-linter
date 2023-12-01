## ENTITY EXTRACTION APPROACH 3:
## Extract entities using spAcy and store summary about
## the sentences relating to those entities
## Benefits: might be more space-efficient
## Shares downsides of spaCy

from memory import Memory
import openai
import spacy
from utils import load
import json

class EntitySummarySpaCyMemory(Memory):
    def __init__(self, source_text, k=5):
        super().__init__(source_text)
        nlp = spacy.load("en_core_web_sm")
        entity_dict = {}

        # Fetch all relevant entities from every line
        # and add them to entity_dict
        for line in source_text.splitlines():
            print("The line is")
            print(line)
            doc = nlp(line)
            for ent in doc.ents:
                # If ent exists, check if we need to summarize it
                if ent.text.upper() in entity_dict:
                    # We need to summarize
                    if entity_dict[ent.text.upper()][2] == k:
                        print("==   time to summarize!   ==")
                        print("The entity is")
                        print(ent.text)
                        summary_so_far = entity_dict[ent.text.upper()][0]
                        new_lines = entity_dict[ent.text.upper()][1] + " " + ent.sent.text
                        prompt = f"""
                            You are summarizing information about an entity.
                            This is the summary so far: \n {summary_so_far}

                            And this is the new lines added about the entity: \n {new_lines}
                            Please return the new summary for the new conversation. Ensure that the summarization is detailed and includes all relevant information about subjects in the conversation. The summary:
                            """
                        response = openai.Completion.create(
                            engine="text-davinci-003",
                            prompt=prompt,
                            temperature=0,
                            max_tokens=250,
                            stop=None,
                            timeout=10
                        )
                        print("THIS IS THE PROPMT")
                        print(prompt)
                        summary = response.choices[0].text.strip()
                        entity_dict[ent.text.upper()][0] = summary
                        entity_dict[ent.text.upper()][1] = ""
                        entity_dict[ent.text.upper()][2] = 0
                    # We don't need to summarize - add to raw sentences
                    else:
                        print("Don't need to summarize, let's add")
                        print(ent.sent.text)
                        print("to the entity")
                        print(ent.text)
                        print("With count")
                        print(entity_dict[ent.text.upper()][2])
                        entity_dict[ent.text.upper()][1] += " " + ent.sent.text
                        entity_dict[ent.text.upper()][2] += 1
                else:
                    # Summary, raw sentences that haven't been summarized yet, count of sentences that haven't been summarized yet
                    entity_dict[ent.text.upper()] = ["", ent.sent.text, 0]

        self.entity_dict = entity_dict
        print("Entity dict")
        print(entity_dict)

    def query(self, query):
        nlp = spacy.load("en_core_web_sm")
        context = ""

        # Find all the relevant entities in the query
        doc = nlp(query)
        for ent in doc.ents:
            # Put existing entities in context
            if ent.text.upper() in self.entity_dict:
                # Fetch both summaries and any overhanging raw sentences to put into context
                context += self.entity_dict[ent.text.upper()][0] + ". "
                context += self.entity_dict[ent.text.upper()][1] + ". "

        # Use context to fetch GPT response
        prompt = f"""You are a smart, knowledgable, accurate AI with the following information:
            {context}
        \nYou are sure about your answers. Please answer the following question: {query}
        """
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=250,
        )
        return response.choices[0].text.strip()

if __name__ == "__main__":
    source_text = load("test.txt")
    memory_test = EntitySummarySpaCyMemory(source_text)

    query ="What is Mimi's favorite physical activity?"
    answer = memory_test.query(query)
    print(query)
    print(answer)

