import re
import pandas as pd
from collections import namedtuple
import cohere
from annoy import AnnoyIndex
import numpy as np

class AIManager:
    def __init__(self, API_KEY):
        pd.set_option('max_colwidth', None)

        self.co = cohere.Client(API_KEY)
        self.create_products()
        self.generate_kb()

 

    def generate_kb(self):
        self.kb = pd.DataFrame({'question': []})

        for product in self.products:
            response = self.co.generate(
                model='xlarge',
                prompt=product.prompt,
                max_tokens=200,
                temperature=0.3,
                k=0,
                p=0.75,
                frequency_penalty=0,
                presence_penalty=0,
                stop_sequences=[],
                return_likelihoods='NONE')
            results = response.generations[0].text
            df = self.generate_df(results)
            self.kb = pd.concat([self.kb, df], axis=0).reset_index(drop=True) 

    def answer_message(self, msg: str, n_top: int = 3) -> list[str]:
        kb_df = self.query_using_semantic_search(msg)
        gen = self.generate_using_dialog(msg)
        result_df = kb_df.append(pd.DataFrame.from_dict({'question': [gen], "distance": [1]}), ignore_index=True)
        return result_df.sort_values("distance")

    def create_products(self):
        product = namedtuple('product', ['name', 'prompt'])

        data = [{'name': 'Halls Standard Cold Frame',
                'prompt': 'Generate questions from this text:  \n\nProduct: Halls Standard Cold Frame\nSturdy Aluminium Framework - rot and rust proof, maintenance free. \n\nAvailable With Two Types Of Glazing - choose from either 3mm Toughened Glass (if broken this glass granulates removing any danger of injury) or Polycarbonate (which is virtually unbreakable). Glazing is for all sides and the top of the cold frame.\n \nDimensions With Toughened Glass :\nWidth – 4ft 3in (129cm)\nDepth – 2ft 1in (63cm)\nHeight at the back – 1ft 3in (38cm) sloping to 1ft 1in (33cm) at the front\n\nDimensions With Polycarbonate :\nWidth – 3ft 3in (99cm)\nDepth – 2ft (60cm)\nHeight at the back – 1ft 4in (40cm) \n \nTwo Sliding, Hinged Lids - allow access to all areas of the cold frame. They also enable you to alter ventilation to your plants.\n\nDelivery - delivered direct from the manufacturers please allow up to 4-6 weeks for delivery.\n--\nQuestion: What is the delivery period for the Halls Standard Cold Frame?\nAnswer: 4-6 weeks\n--\nQuestion: What is the width in cm for the Halls Standard toughened glass Cold Frame?\nAnswer: 129cm\n--\nQuestion: What is the depth for the toughened glass in feet and inches for the Halls Standard Cold Frame?\nAnswer: 2ft 1in\n--\nQuestion: What is the height at the back in cm for the Halls Standard Polycarbonate Cold Frame?\nAnswer: 40cm\n--\nQuestion: What is the height at the front in feet and inches for the Halls Standard toughened glass Cold Frame?\nAnswer: 1ft 1in\n--\nQuestion: What is the width for the polycarbonate in cm for the Halls Standard Cold Frame?\nAnswer: 99cm\n--\nQuestion: What is the depth for the polycarbonate in feet and inches for the Halls Standard Cold Frame?\nAnswer: 2ft\n--\nQuestion: What is the height at the back in cm for the Halls Standard Cold Frame?\nAnswer: 1ft 4in\n--\nQuestion: What is the height at the front in cm for the Halls Standard Cold Frame?\nAnswer: 1ft 1in\n--\nQuestion: What is the height at the back in cm for the Halls Standard Cold Frame?\nAnswer: 1\n--\n'},
                {'name': 'Rowlinson Timber Coldframe',
                'prompt': 'Generate questions from this text:  \n\nProduct: Rowlinson Timber Coldframe\n\nFSC Pressure Treated 19mm Softwood Frame - manufactured from FSC certified timber from sustainable sources. It has been pressure treated against rot. You can stain or paint the frame to match your garden if required. \n \nTwo Independently Opening Lids - allowing easy access to the plants in your cold frame. Supplied complete with wooden stays, with two height setting, to allow excellent ventilation. The lid is glazed with clear styrene plastic, allowing excellent light transmission and is virtually unbreakable.\n\nDimensions :\nWidth - 3ft 4in / 102cm \nDepth - 2ft 8in / 81cm \nHeight at back - 1ft 3in / 38cm\nHeight at front - 11in / 29cm\n\nSelf Assembly\nThis cold frame is delivered as pre assembled panels which simply need screwing together. The lid is supplied fully glazed and should be screwed into place together with the stays provided. You will need a cross-head screwdriver during construction.\n\nDelivery : please allow up to 14 working days for delivery.\n--\nQuestion: What is the delivery period for the Rowlinson Timber Cold Frame?\nAnswer: Up to 14 working days\n--\nQuestion: What is the width in inches for the Rowlinson Timber Cold Frame?\nAnswer: 3ft 4in\n--\nQuestion: What is the height at the back in cm for the Rowlinson Timber Cold Frame?\nAnswer: 38cm\n--\n'},
                {'name': 'Haxnicks Grower Frame Polythene Cover',
                'prompt': 'Generate questions from this text:  \n\nProduct: Haxnicks Grower Frame Polythene Cover\n\nShaped to easily fit over the Grower Frame to create a protected space that will retain warmth and humidity for quicker plant growth.\nFour zips on the sides of the cover lets you easily access all areas of the area under cover.\nRoll up insect proof ventilation panels at either end of the cover allow air to circulate whilst preventing insects from getting to your plants.\nSize: 9’8\" long x 3’3\" wide x 3’3\" high (3 metres x 1 metre x 1 metre)\n--\nQuestion: How long is the Haxnicks Grower Frame Polythene Cover in feet and inches?\nAnswer: 9’8\"\n--\nQuestion: What is the width of the Haxnicks Grower Frame Polythene Cover in metres?\nAnswer: 1 metre\n--\nQuestion: How high is the Haxnicks Grower Frame Polythene Cover in feet and inches?\nAnswer: 3’3\"\n--\n'},] 

        self.products = [product(**item) for item in data] 

    def generate_df(self, results):
        question = []
        answer = []
        results = re.sub('\n',' ', results)
        results = results.split('--')
        results = [result.strip() for result in results]
        for result in results:
            if 'Question' in result:
                out = re.findall(r'Question: (.*?)? Answer: (.*?)$',result)
                for item in out:
                    if item:
                        q, a = item
                        question.append(q + ' ' + a)
        return pd.DataFrame({'question': question}) 

    def query_using_semantic_search(self, query):
        df = self.kb
        embeds = self.co.embed(texts=list(df['question']),
                            model="large",
                            truncate="LEFT").embeddings
        embeds = np.array(embeds)
        num_entries, num_dimensions = embeds.shape

        search_index = AnnoyIndex(num_dimensions, 'angular')

        for i in range(len(embeds)):
            search_index.add_item(i, embeds[i])

        search_index.build(10) 
        search_index.save('test.ann')

        query_embed = self.co.embed(texts=[query],
                                model='large',
                                truncate='LEFT').embeddings
        similar_item_ids = search_index.get_nns_by_vector(query_embed[0],
                                                            2,
                                                            include_distances=True)
        return pd.DataFrame({'question': df.loc[similar_item_ids[0], 'question'],
                            'distance': similar_item_ids[1]}) 

    def generate_using_dialog(self, dialog):
        promt_text = f"""You are a customer support agent responding to a customer.
--
Customer: Hello.
Agent: Hello, what can I help you with today?
--
Customer: {dialog}
Agent:"""

        response = self.co.generate(
            model='xlarge',
            prompt=promt_text,
            max_tokens=15,
            temperature=0.3,
            k=0,
            p=0.75,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=["--"],
            return_likelihoods='NONE')
        return  response.generations[0].text.split("--")[0].strip()



if __name__ == "__main__":
    API_KEY = 'bULA1eGPpCKwDioqSK49DrepiSSHuvRQ8gTwKjAs'
    aiManager = AIManager(API_KEY)
    msg = 'What is the height at the back in cm for the Halls Standard Cold Frame'
    response = aiManager.answer_message(msg)
    print(response)
